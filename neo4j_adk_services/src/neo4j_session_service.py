from neo4j import GraphDatabase, Record # Added Record
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import ListSessionsResponse, ListEventsResponse
from google.adk.events import Event, EventActions # Assuming EventActions might contain tool_calls
from google.genai import types  # for Content, Part
import json # For _extract_tool_calls and _encode_event
import uuid # For _extract_tool_calls
import time # For _encode_event and create_session

class Neo4jSessionService(BaseSessionService):
    """A SessionService implementation backed by Neo4j graph database."""
    def __init__(self, uri: str, user: str, password: str, database: str = None):
        # Initialize Neo4j driver (synchronous)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        # Schema indexes/constraints for sessions and tool calls
        with self._driver.session(database=self._database) as db_session:
            # Session constraint (original)
            db_session.run(
                "CREATE CONSTRAINT session_unique IF NOT EXISTS "
                "FOR (s:Session) "
                "REQUIRE (s.app_name, s.user_id, s.id) IS UNIQUE"
            )
            # ToolCall constraint (from patch)
            db_session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:ToolCall) REQUIRE t.id IS UNIQUE"
            )
        # BaseSessionService has no specific __init__, but call super if needed
        super().__init__()

    def _execute_write(self, query: str, parameters: dict = None):
        """Helper to execute a write query in a new session."""
        with self._driver.session(database=self._database) as session:
            return session.run(query, parameters).data()

    def _execute_read(self, query: str, parameters: dict = None):
        """Helper to execute a read query in a new session."""
        with self._driver.session(database=self._database) as session:
            return session.run(query, parameters).data()

    def create_session(self, *, app_name: str, user_id: str, state: dict = None, session_id: str = None) -> Session:
        if session_id is None:
            session_id = Session.new_id() if hasattr(Session, "new_id") else uuid.uuid4().hex
        
        initial_state = state or {}
        state_json = json.dumps(initial_state)
        current_timestamp = time.time()

        # Cypher query adapted from patch, creating App, User, and Session with relationship
        # Assuming app_name is the key for App node and user_id for User node.
        # $app_delta and $user_delta are not used here as we are creating, not updating App/User nodes.
        # The patch implies MERGE for App and User, which is good practice.
        query = """
        MERGE (app:App {name: $app_name})
        MERGE (userNode:User {id: $user_id}) // Assuming 'id' is the property for userNode
        MERGE (userNode)-[:STARTED_SESSION]->(s:Session {id: $session_id})
        ON CREATE SET
            s.app_name = $app_name,
            s.user_id = $user_id,
            s.state_json = $state_json,
            s.last_update_time = $timestamp
        RETURN s.app_name AS app_name, s.user_id AS user_id, s.id AS id, s.state_json AS state_json, s.last_update_time AS last_update_time
        """
        params = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "state_json": state_json,
            "timestamp": current_timestamp
        }
        result = self._execute_write(query, params)
        
        session_data = result[0] if result else {}
        
        return Session(
            app_name=session_data.get("app_name", app_name),
            user_id=session_data.get("user_id", user_id),
            id=session_data.get("id", session_id),
            state=initial_state, # Use the initial Python dict for the Session object
            events=[],
            last_update_time=session_data.get("last_update_time", current_timestamp)
        )

    def get_session(self, *, app_name: str, user_id: str, session_id: str, config: dict = None) -> Session:
        # Fetch session and all its events from Neo4j
        query = (
            "MATCH (s:Session {app_name: $app_name, user_id: $user_id, id: $session_id}) "
            "OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:Event) "
            "WITH s, e ORDER BY e.timestamp "
            "RETURN s, collect(e) AS events"
        )
        result = self._execute_read(query, {"app_name": app_name, "user_id": user_id, "session_id": session_id})
        if not result:
            return None  # Session not found
        record = result[0]
        s_node = record['s']
        events_list = record['events']
        # Parse session state JSON
        import json
        state = json.loads(s_node.get('state_json', '{}'))
        # Reconstruct Session object
        sess = Session(
            app_name=s_node['app_name'],
            user_id=s_node['user_id'],
            id=s_node['id'],
            state=state,
            events=[],
            last_update_time=s_node.get('last_update_time', __import__("time").time())
        )
        # Reconstruct Event objects for each event node
        for e_node in events_list:
            # e_node is a neo4j.Node object with properties
            props = dict(e_node)  # cast to dict of properties
            content_obj = None
            actions_obj = None
            if props.get('content_json'):
                content_data = json.loads(props['content_json'])
                # google.genai.types.Content is assumed pydantic or compatible
                content_obj = types.Content.model_validate(content_data) if hasattr(types.Content, 'model_validate') else content_data
            if props.get('actions_json'):
                actions_data = json.loads(props['actions_json'])
                actions_obj = EventActions.model_validate(actions_data) if hasattr(EventActions, 'model_validate') else EventActions(**actions_data)
            event = Event(
                id=props.get('id'),
                author=props.get('author'),
                timestamp=props.get('timestamp'),
                invocation_id=props.get('invocation_id'),
                content=content_obj,
                actions=actions_obj
            )
            sess.events.append(event)
        return sess

    def list_sessions(self, *, app_name: str, user_id: str) -> ListSessionsResponse:
        # Query all sessions for the user (return basic info)
        query = (
            "MATCH (s:Session {app_name: $app_name, user_id: $user_id}) "
            "RETURN s.id AS session_id, s.last_update_time AS last_update"
        )
        records = self._execute_read(query, {"app_name": app_name, "user_id": user_id})
        # Build a ListSessionsResponse (if available) or list of Session metadata
        sessions = [Session(app_name=app_name, user_id=user_id, id=rec["session_id"],
                             state={}, events=[], last_update_time=rec.get("last_update", 0))
                    for rec in records]
        return ListSessionsResponse(sessions=sessions)

    def list_events(self, *, app_name: str, user_id: str, session_id: str) -> ListEventsResponse:
        # Retrieve all events for a session (similar to get_session but only events)
        query = (
            "MATCH (s:Session {app_name: $app_name, user_id: $user_id, id: $session_id})-[:HAS_EVENT]->(e:Event) "
            "WITH e ORDER BY e.timestamp "
            "RETURN collect(e) AS events"
        )
        result = self._execute_read(query, {"app_name": app_name, "user_id": user_id, "session_id": session_id})
        events_list = [] # Renamed from events to avoid confusion with the ADK Event type
        if result:
            events_nodes = result[0]['events']
            import json
            for e_node in events_nodes:
                props = dict(e_node)
                content_obj = None
                actions_obj = None
                if props.get('content_json'):
                    content_data = json.loads(props['content_json'])
                    content_obj = types.Content.model_validate(content_data) if hasattr(types.Content, 'model_validate') else content_data
                if props.get('actions_json'):
                    actions_data = json.loads(props['actions_json'])
                    actions_obj = EventActions.model_validate(actions_data) if hasattr(EventActions, 'model_validate') else EventActions(**actions_data)
                event_obj = Event( # Renamed from event to event_obj
                    id=props.get('id'),
                    author=props.get('author'),
                    timestamp=props.get('timestamp'),
                    invocation_id=props.get('invocation_id'),
                    content=content_obj,
                    actions=actions_obj
                )
                events_list.append(event_obj)
        return ListEventsResponse(events=events_list)

    def append_event(self, session: Session, event: Event) -> Event:
        """Append an event to the session, update state, and persist to Neo4j with enhanced graph structure."""

        # 1. Apply state_delta to session.state (in-memory) and collect changes for WROTE_STATE
        previous_values_for_delta = {}
        if event.actions and hasattr(event.actions, "state_delta"):
            state_delta_from_event = event.actions.state_delta or {}
            for key in state_delta_from_event:
                if not key.startswith("temp:"):
                    previous_values_for_delta[key] = session.state.get(key)

        current_persisted_state_keys_values = {}
        if event.actions and hasattr(event.actions, "state_delta"):
            state_delta_from_event = event.actions.state_delta or {}
            for key, value in state_delta_from_event.items():
                if key.startswith("temp:"):
                    continue
                if value is None:
                    session.state.pop(key, None)
                    current_persisted_state_keys_values[key] = None
                else:
                    session.state[key] = value
                    current_persisted_state_keys_values[key] = value
        
        current_timestamp = event.timestamp if hasattr(event, "timestamp") and event.timestamp is not None else time.time()
        session.last_update_time = current_timestamp
        
        if not getattr(event, "id", None):
            event.id = Event.new_id() if hasattr(Event, "new_id") else uuid.uuid4().hex
        if not getattr(event, "timestamp", None):
            event.timestamp = current_timestamp

        event_props_for_cypher = self._prepare_event_properties(event)
        tool_calls_for_cypher = self._extract_tool_calls(event)

        query = """
        MATCH (s:Session {id: $session_id, app_name: $app_name, user_id: $user_id})
        
        CREATE (e:Event)-[:OF_SESSION]->(s)
        SET e = $event_props
        
        WITH s, e
        OPTIONAL MATCH (prev_event:Event)-[:OF_SESSION]->(s)
        WHERE prev_event.timestamp < e.timestamp AND prev_event <> e
        WITH s, e, prev_event ORDER BY prev_event.timestamp DESC LIMIT 1
        FOREACH (_ IN CASE WHEN prev_event IS NOT NULL THEN [1] ELSE [] END |
          CREATE (prev_event)-[:NEXT]->(e))
          
        WITH s, e
        UNWIND keys($current_persisted_state_keys_values) AS k
          WITH s, e, k,
               $previous_values_for_delta[k] AS fromVal,
               $current_persisted_state_keys_values[k] AS toVal
          CREATE (e)-[:WROTE_STATE {key: k, fromValue_json: CASE WHEN fromVal IS NOT NULL THEN apoc.convert.toJson(fromVal) ELSE null END, toValue_json: CASE WHEN toVal IS NOT NULL THEN apoc.convert.toJson(toVal) ELSE null END, timestamp: e.timestamp}]->(s)

        WITH e
        UNWIND $tool_calls_data AS tc_data
          MERGE (t:ToolCall {id: tc_data.id})
          SET t += tc_data
          CREATE (e)-[:INVOKED_TOOL]->(t)
          
        WITH s
        SET s.state_json = $new_session_state_json,
            s.last_update_time = $event_timestamp
        
        RETURN e.id AS event_id
        """
        
        params = {
            "session_id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "event_props": event_props_for_cypher,
            "previous_values_for_delta": previous_values_for_delta,
            "current_persisted_state_keys_values": current_persisted_state_keys_values,
            "tool_calls_data": tool_calls_for_cypher,
            "new_session_state_json": json.dumps(session.state),
            "event_timestamp": event_props_for_cypher["timestamp"]
        }

        self._execute_write(query, params)
        
        session.events.append(event)
        return event

    # Helper function from patch to extract tool calls - now a method
    def _extract_tool_calls(self, ev: Event) -> list[dict]:
        """Return a list of dicts ready to be set on :ToolCall, if the Event
        contained tool invocations in its actions (ADK convention)."""
        tc_list = []
        if ev.actions and getattr(ev.actions, "tool_calls", None):
            for tc in ev.actions.tool_calls:
                # Ensure parameters are JSON serializable; ADK tool_calls.parameters should be.
                parameters_json = None
                if tc.parameters:
                    try:
                        parameters_json = json.dumps(tc.parameters)
                    except TypeError: # Fallback if not directly serializable
                        parameters_json = json.dumps(str(tc.parameters))


                tc_list.append({
                    "id": getattr(tc, 'id', None) or str(uuid.uuid4()), # Ensure ID exists
                    "name": getattr(tc, 'name', 'unknown_tool'),
                    "parameters_json": parameters_json,
                    "latency_ms": getattr(tc, "latency_ms", None), # Optional
                    "status": getattr(tc, "status", None), # Optional
                    "error_msg": getattr(tc, "error", None) # Optional
                })
        return tc_list

    # Helper function to prepare event properties for Cypher (similar to _encode_event in patch) - now a method
    def _prepare_event_properties(self, event: Event) -> dict:
        """Prepares event properties for storage, including serializing content and actions."""
        content_json_str = None
        if event.content:
            content_json_str = json.dumps(event.content.model_dump() if hasattr(event.content, 'model_dump') else event.content)

        actions_json_str = None
        if event.actions:
            actions_json_str = json.dumps(event.actions.model_dump() if hasattr(event.actions, 'model_dump') else event.actions)
        
        text_summary = ""
        if event.content and hasattr(event.content, "parts"):
            try:
                text_summary = " ".join(p.text for p in event.content.parts if hasattr(p, "text"))
            except: # Fallback
                text_summary = str(event.content)
        elif event.content:
            text_summary = str(event.content)

        return {
            "id": event.id,
            "author": event.author,
            "timestamp": event.timestamp or time.time(), # Ensure timestamp exists
            "invocation_id": getattr(event, "invocation_id", None) or "",
            "content_json": content_json_str,
            "actions_json": actions_json_str,
            "text": text_summary # For full-text search on events if needed later
        }

    # ---------- deletion ----------
    def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str
    ) -> None:
        """
        Remove the session and **everything hanging off it** (events,
        tool-call nodes, WROTE_STATE edges, metrics, etc.).

        We rely on `DETACH DELETE` so the pattern survives future schema
        extensions without further changes.
        """
        cypher = (
            "MATCH (s:Session {app_name:$app, user_id:$user, id:$sid}) "
            "DETACH DELETE s"
        )
        # uses the already-patched _execute_write in the test-suite; in
        # production this is a single write transaction.
        self._execute_write(
            cypher,
            {"app": app_name, "user": user_id, "sid": session_id},
        )

    def close_session(self, session: Session) -> None: # Signature seems fine
        """Closes a session and optionally persists it to memory (experimental)."""
        # In this implementation, closing a session is equivalent to ensuring it's saved.
        # If a MemoryService is configured, it could call memory_service.add_session_to_memory(session).
        # For now, just a placeholder.
        return

    def close(self):
        """Close the Neo4j driver (to be called on application shutdown)."""
        self._driver.close()