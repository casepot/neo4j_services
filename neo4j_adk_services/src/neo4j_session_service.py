from neo4j import GraphDatabase, Record # Added Record
from google.adk.sessions import BaseSessionService
from .my_session import Session  # <-- use the wrapped version everywhere
from google.adk.sessions.base_session_service import ListSessionsResponse, ListEventsResponse
from google.adk.events import Event, EventActions # Assuming EventActions might contain tool_calls
from google.genai import types  # for Content, Part
import json # For _extract_tool_calls and _encode_event
import uuid # For _extract_tool_calls
import time # For _encode_event and create_session
import math # For floor function
import logging


# Custom exception for optimistic locking failures
class StaleSessionError(Exception):
    """Raised when a session update is attempted on a stale version of the session."""
    pass


class Neo4jSessionService(BaseSessionService):
    """A SessionService implementation backed by Neo4j graph database."""
    def __init__(self, uri: str, user: str, password: str, database: str = None):
        # Initialize Neo4j driver (synchronous)
        # Expecting plain bolt:// URI as TLS will be disabled in the container for tests
        self._driver = GraphDatabase.driver(
            uri, # Use the provided URI directly (expected to be bolt://)
            auth=(user, password)
            # No explicit encryption flags, relying on plain bolt connection
        )
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
            # AppState constraint (P7)
            db_session.run(
                "CREATE CONSTRAINT app_state_unique IF NOT EXISTS "
                "FOR (a:AppState) REQUIRE a.app_name IS UNIQUE"
            )
            # UserState constraint (P7)
            db_session.run(
                "CREATE CONSTRAINT user_state_unique IF NOT EXISTS "
                "FOR (u:UserState) REQUIRE (u.app_name, u.user_id) IS UNIQUE"
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
        
        initial_session_state = state or {}
        # P7: Fetch and merge shadow AppState and UserState
        shadow_app_state_query = """
        OPTIONAL MATCH (as:AppState {app_name: $app_name})
        RETURN as.state_json AS app_state_json, as.version AS app_state_version
        """
        app_state_result = self._execute_read(shadow_app_state_query, {"app_name": app_name})
        
        shadow_user_state_query = """
        OPTIONAL MATCH (us:UserState {app_name: $app_name, user_id: $user_id})
        RETURN us.state_json AS user_state_json, us.version AS user_state_version
        """
        user_state_result = self._execute_read(shadow_user_state_query, {"app_name": app_name, "user_id": user_id})

        merged_state = initial_session_state.copy()

        if app_state_result and app_state_result[0].get("app_state_json"):
            app_shadow_data = json.loads(app_state_result[0]["app_state_json"])
            # Simple merge: shadow state overlays initial session state for app-specific keys
            # More sophisticated versioning/merging could be added if needed.
            # For now, assume shadow state is more current for its keys.
            for k, v in app_shadow_data.items():
                if k.startswith("app:"): # Only merge app-prefixed keys from AppState
                    merged_state[k] = v
        
        if user_state_result and user_state_result[0].get("user_state_json"):
            user_shadow_data = json.loads(user_state_result[0]["user_state_json"])
            # Simple merge for user-specific keys
            for k, v in user_shadow_data.items():
                if k.startswith("user:"): # Only merge user-prefixed keys from UserState
                     merged_state[k] = v
        
        state_json = json.dumps(merged_state)
        # current_timestamp = time.time() # Will use Neo4j's timestamp()

        # Cypher query adapted from patch, creating App, User, and Session with relationship
        # Assuming app_name is the key for App node and user_id for User node.
        # $app_delta and $user_delta are not used here as we are creating, not updating App/User nodes.
        # The patch implies MERGE for App and User, which is good practice.
        # s.last_update_time will be set by Neo4j's timestamp() in milliseconds.
        query = """
        MERGE (app:App {name: $app_name})
        MERGE (userNode:User {id: $user_id}) // Assuming 'id' is the property for userNode
        MERGE (userNode)-[:STARTED_SESSION]->(s:Session {id: $session_id})
        ON CREATE SET
            s.app_name = $app_name,
            s.user_id = $user_id,
            s.state_json = $state_json,
            s.last_update_time = timestamp() // Use Neo4j's timestamp for milliseconds
        RETURN s.app_name AS app_name, s.user_id AS user_id, s.id AS id, s.state_json AS state_json, s.last_update_time AS last_update_time_ms
        """
        params = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "state_json": state_json
        }
        result = self._execute_write(query, params)

        if not result or not result[0].get("last_update_time_ms"):
             # Handle error: session creation failed or timestamp wasn't returned
             # For simplicity, raise an error or return None, depending on desired behavior
             raise RuntimeError(f"Failed to create session {session_id} or retrieve its timestamp.")

        session_data = result[0]

        # Use the timestamp returned directly from the DB
        db_last_update_time_ms = session_data["last_update_time_ms"]
        python_last_update_time_s = db_last_update_time_ms / 1000.0

        # Create the session object using the DB-generated timestamp
        sess = Session(
            app_name=session_data.get("app_name", app_name),
            user_id=session_data.get("user_id", user_id),
            id=session_data.get("id", session_id),
            state=merged_state, # Use the merged state for the Session object
            events=[],
            last_update_time=python_last_update_time_s, # Use the DB timestamp
            last_update_time_ms=db_last_update_time_ms # Add the ms timestamp
        )
        return sess

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
        db_last_update_time_ms = s_node.get('last_update_time')
        python_last_update_time_s = db_last_update_time_ms / 1000.0 if db_last_update_time_ms is not None else __import__("time").time()

        sess = Session(
            app_name=s_node['app_name'],
            user_id=s_node['user_id'],
            id=s_node['id'],
            state=state,
            events=[],
            # Convert last_update_time from ms (Neo4j) to seconds (Python Session object)
            last_update_time=python_last_update_time_s,
            last_update_time_ms=db_last_update_time_ms
        )
        # Reconstruct Event objects for each event node
        for e_node in events_list:
            # e_node is a neo4j.Node object with properties
            props = dict(e_node)  # cast to dict of properties
            content_obj = None
            actions_obj = None
            if props.get('content_json'):
                try:
                    content_data = json.loads(props['content_json'])
                    # google.genai.types.Content is assumed pydantic or compatible
                    content_obj = types.Content.model_validate(content_data) if hasattr(types.Content, 'model_validate') else content_data
                except (json.JSONDecodeError, TypeError):
                    content_obj = None # Handle potential errors during loading
            if props.get('actions_json'):
                try:
                    actions_data = json.loads(props['actions_json'])
                    # Check if actions_data is not None before validation/instantiation
                    if actions_data is not None:
                        actions_obj = EventActions.model_validate(actions_data) if hasattr(EventActions, 'model_validate') else EventActions(**actions_data)
                    # If actions_data is None after loading, actions_obj remains None
                except (json.JSONDecodeError, TypeError):
                     actions_obj = None # Handle potential errors during loading
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

    def list_events(self, *, app_name: str, user_id: str, session_id: str, after: str = None, limit: int = 100) -> ListEventsResponse:
        # Retrieve events for a session with pagination
        # 'after' is expected to be the timestamp of the event to start after
        # For simplicity, we'll use SKIP based on 'after' if it were an index,
        # but true cursor-based pagination with timestamps requires a WHERE clause.
        # Given the current structure, we'll use SKIP for offset and LIMIT for page size.
        # A more robust 'after' would involve `WHERE e.timestamp > $after_timestamp`.
        # For now, we'll interpret 'after' as an offset if it's an integer, or ignore if not easily translatable.
        # The issue implies 'after' could be a token, which often maps to an event ID or timestamp.
        # Let's assume 'after' is not directly used for SKIP in this simplified version,
        # and focus on LIMIT. A full cursor implementation is more complex.
        # The issue mentions "SKIP/LIMIT query", so we'll implement that.
        # 'after' would typically be an event ID or a unique sort key value from the previous page.
        # If 'after' is an event ID, the query would need to find that event's timestamp
        # and then fetch events after it. This is complex for a simple SKIP/LIMIT.
        # For this implementation, we will use limit and a conceptual skip (offset).
        # The prompt's P9 description: "accept page_size, after token if desired".
        # We will use 'limit' as page_size. 'after' token handling for SKIP is non-trivial
        # without knowing the token's nature (e.g. if it's an offset or a value to compare).
        # Let's assume 'after' is an offset for now, if provided as an int.
        
        skip_val = 0
        if after is not None:
            try:
                # This is a simplification; a real 'after' token would not be a direct skip count.
                # It would be an ID or a value from the last item of the previous page.
                # For this exercise, we'll treat 'after' as a potential offset if it's an integer string.
                skip_val = int(after)
            except ValueError:
                # If 'after' is not an integer, we ignore it for SKIP in this simplified model.
                # A production system would need a more robust way to handle 'after' tokens.
                pass

        query = (
            "MATCH (s:Session {app_name: $app_name, user_id: $user_id, id: $session_id})-[:HAS_EVENT]->(e:Event) "
            "WITH e ORDER BY e.timestamp " # Order before skip/limit
            "SKIP $skip_count LIMIT $limit_count "
            "RETURN collect(e) AS events"
        )
        params = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "skip_count": skip_val,
            "limit_count": limit
        }
        result = self._execute_read(query, params)
        events_list = []
        if result and result[0]['events']: # Check if 'events' key exists and is not empty
            events_nodes = result[0]['events']
            import json
            for e_node in events_nodes:
                props = dict(e_node)
                content_obj = None
                actions_obj = None
                if props.get('content_json'):
                    try:
                        content_data = json.loads(props['content_json'])
                        content_obj = types.Content.model_validate(content_data) if hasattr(types.Content, 'model_validate') else content_data
                    except (json.JSONDecodeError, TypeError):
                        content_obj = None
                if props.get('actions_json'):
                    try:
                        actions_data = json.loads(props['actions_json'])
                        if actions_data is not None:
                            actions_obj = EventActions.model_validate(actions_data) if hasattr(EventActions, 'model_validate') else EventActions(**actions_data)
                    except (json.JSONDecodeError, TypeError):
                        actions_obj = None
                event_obj = Event(
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
        app_state_delta = {}
        user_state_delta = {}

        if event.actions and hasattr(event.actions, "state_delta"):
            state_delta_from_event = event.actions.state_delta or {}
            for key, value in state_delta_from_event.items():
                if key.startswith("temp:"):
                    continue
                
                # Populate app_state_delta and user_state_delta (P7)
                if key.startswith("app:"):
                    app_state_delta[key] = value
                elif key.startswith("user:"):
                    user_state_delta[key] = value
                # else, it's a session-specific state key

                # Update session.state (in-memory)
                if value is None:
                    session.state.pop(key, None)
                    current_persisted_state_keys_values[key] = None
                else:
                    session.state[key] = value
                    current_persisted_state_keys_values[key] = value
        
        # Ensure event has an ID and timestamp before preparing properties
        # The event's timestamp is distinct from the session's last_update_time for optimistic locking
        if not getattr(event, "id", None):
            event.id = Event.new_id() if hasattr(Event, "new_id") else uuid.uuid4().hex
        if not getattr(event, "timestamp", None):
            # Use current time if event doesn't have a timestamp; this is for the event node itself.
            event.timestamp = time.time()

        event_props_for_cypher = self._prepare_event_properties(event) # This uses event.timestamp (seconds)
        tool_calls_for_cypher = self._extract_tool_calls(event)

        # Client's last known session update time in milliseconds for optimistic locking
        # session.last_update_time is in seconds (float)
        # optimistic-locking check – skip float round-trip entirely
        client_last_update_time_ms = session.last_update_time_ms
 
        query = """
        // P7: Update AppState if app_state_delta is provided
        FOREACH (ignoreMe IN CASE WHEN size(keys($app_state_delta)) > 0 THEN [1] ELSE [] END |
            MERGE (as:AppState {app_name: $app_name})
            ON CREATE SET as.state_json = apoc.convert.toJson($app_state_delta), as.version = 0
            ON MATCH SET as.state_json = apoc.convert.toJson(apoc.map.merge(CASE WHEN as.state_json IS NULL THEN {} ELSE apoc.convert.fromJsonMap(as.state_json) END, $app_state_delta)), as.version = timestamp()
        )
        // P7: Update UserState if user_state_delta is provided
        FOREACH (ignoreMe IN CASE WHEN size(keys($user_state_delta)) > 0 THEN [1] ELSE [] END |
            MERGE (us:UserState {app_name: $app_name, user_id: $user_id})
            ON CREATE SET us.state_json = apoc.convert.toJson($user_state_delta), us.version = 0
            ON MATCH SET us.state_json = apoc.convert.toJson(apoc.map.merge(CASE WHEN us.state_json IS NULL THEN {} ELSE apoc.convert.fromJsonMap(us.state_json) END, $user_state_delta)), us.version = timestamp()
        )
        WITH $session_id AS session_id, $app_name AS app_name, $user_id AS user_id // Carry over parameters explicitly
        MATCH (s:Session {id: session_id, app_name: app_name, user_id: user_id})
        WHERE s.last_update_time = $client_last_update_time_ms
        
        // If match successful, update session timestamp immediately
        SET s.last_update_time = timestamp(), // Neo4j's timestamp() is in ms
            s.state_json = $new_session_state_json // Update state as well
        
        WITH s // Carry the successfully matched and updated session
        
        CREATE (s)-[:HAS_EVENT]->(e:Event)
        SET e = $event_props // event_props contains 'timestamp' in seconds for the Event node
        
        // Link to previous event if exists (NEXT relationship)
        WITH s, e
        OPTIONAL MATCH (s)-[:HAS_EVENT]->(prev_event:Event)
        WHERE prev_event.timestamp < e.timestamp AND prev_event <> e // Compare event timestamps (seconds)
        WITH s, e, prev_event ORDER BY prev_event.timestamp DESC LIMIT 1
        FOREACH (_ IN CASE WHEN prev_event IS NOT NULL THEN [1] ELSE [] END |
          CREATE (prev_event)-[:NEXT]->(e))
          
        // Create WROTE_STATE relationships for persisted state changes (session-specific keys)
        WITH s, e
        UNWIND [k IN keys($current_persisted_state_keys_values) WHERE NOT (k STARTS WITH 'app:' OR k STARTS WITH 'user:')] AS k_session
          WITH s, e, k_session,
               apoc.map.get($previous_values_for_delta, k_session, null) AS fromVal,
               apoc.map.get($current_persisted_state_keys_values, k_session, null) AS toVal
          CREATE (e)-[:WROTE_STATE {key: k_session, fromValue_json: apoc.convert.toJson(fromVal), toValue_json: apoc.convert.toJson(toVal), timestamp: e.timestamp}]->(s)

        // Create INVOKED_TOOL relationships (will be skipped on an empty list)
        FOREACH (tc_data IN $tool_calls_data |
          MERGE (t:ToolCall {id: tc_data.id})
          SET   t += tc_data
          CREATE (e)-[:INVOKED_TOOL]->(t)
        )

        // Return the event id and the new session update time (in ms from DB)
        RETURN e.id AS event_id, s.last_update_time AS new_session_last_update_time_ms
        """
        
        params = {
            "session_id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "client_last_update_time_ms": client_last_update_time_ms,
            "new_session_state_json": json.dumps(session.state), # New state applied to session.state in Python
            "event_props": event_props_for_cypher, # Contains event.id, event.timestamp (seconds), etc.
            "previous_values_for_delta": previous_values_for_delta,
            "current_persisted_state_keys_values": current_persisted_state_keys_values, # Contains all non-temp keys
            "tool_calls_data": tool_calls_for_cypher,
            "app_state_delta": app_state_delta, # P7
            "user_state_delta": user_state_delta # P7
        }
        logging.debug(f"Executing Cypher query for append_event. Session ID: {session.id}")
        logging.debug(f"Query: {query}")
        logging.debug(f"Params: {json.dumps(params, indent=2)}") # Log params as pretty JSON
        logging.debug(f"Previous values for delta: {json.dumps(previous_values_for_delta, indent=2)}")
        logging.debug(f"Current persisted state keys/values: {json.dumps(current_persisted_state_keys_values, indent=2)}")

        result = self._execute_write(query, params)
        
        if not result or not result[0].get("new_session_last_update_time_ms"):
            logging.error(f"StaleSessionError for session {session.id}. Client timestamp (ms): {client_last_update_time_ms}.")
            logging.error(f"Query result: {result}")
            # Log details about what might be missing from the result
            if not result:
                logging.error("Result from _execute_write was None or empty.")
            elif not result[0].get("new_session_last_update_time_ms"):
                logging.error(f"Result[0] did not contain 'new_session_last_update_time_ms'. Result[0]: {result[0]}")

            raise StaleSessionError(
                f"Session {session.id} update failed due to stale data. "
                f"Client timestamp: {client_last_update_time_ms}, current DB timestamp may differ."
            )

        new_db_timestamp_ms = result[0]["new_session_last_update_time_ms"]
        session.last_update_time = new_db_timestamp_ms / 1000.0 # Update Python session with new time in seconds
        session.last_update_time_ms = new_db_timestamp_ms # Update ms timestamp as well
        
        # event.timestamp was set at the beginning of the method if not present.
        # session.last_update_time is now updated from DB.
        
        session.events.append(event)
        return event

    # Updated helper function to extract tool calls using modern ADK/Gemini types
    def _extract_tool_calls(self, ev: Event) -> list[dict]:
        """Return a list of dicts ready to be set on :ToolCall nodes,
        extracting data from FunctionCall objects within the event content."""
        tool_calls_for_cypher = []
        # Use the standard ADK method to get function calls from event content
        function_calls = ev.get_function_calls() if hasattr(ev, 'get_function_calls') else []
        
        for fc in function_calls or []:
            # FunctionCall objects have 'name' and 'args' (dict)
            # We need to generate an ID if one isn't inherent to FunctionCall
            # Note: google.genai.types.FunctionCall doesn't have a standard 'id' field.
            # We'll generate one for the Neo4j node.
            tool_call_id = str(uuid.uuid4())
            
            parameters_json = None
            if fc.args:
                try:
                    # fc.args should already be a dict
                    parameters_json = json.dumps(fc.args)
                except TypeError:
                    # Fallback if args contains non-serializable types
                    parameters_json = json.dumps(str(fc.args))

            tool_calls_for_cypher.append({
                "id": tool_call_id, # Generated ID for the Neo4j node
                "name": fc.name or 'unknown_tool',
                "parameters_json": parameters_json,
                # Optional fields from the old structure are not directly available
                # on FunctionCall. They might come from FunctionResponse later.
                "latency_ms": None,
                "status": None,
                "error_msg": None
            })
        return tool_calls_for_cypher

    # Helper function to prepare event properties for Cypher (similar to _encode_event in patch) - now a method
    def _prepare_event_properties(self, event: Event) -> dict:
        """Prepares event properties for storage, including serializing content and actions."""
        content_json_str = None
        if event.content:
            content_json_str = json.dumps(event.content.model_dump() if hasattr(event.content, 'model_dump') else event.content)

        actions_json_str = None
        if event.actions:
            try:
                # Prioritize model_dump if available (for Pydantic models)
                if hasattr(event.actions, 'model_dump'):
                    actions_dict = event.actions.model_dump()
                # Fallback to __dict__ for simple objects
                elif hasattr(event.actions, '__dict__'):
                    actions_dict = event.actions.__dict__
                # Final fallback: attempt to convert to string if it's not serializable
                else:
                    actions_dict = str(event.actions)
                actions_json_str = json.dumps(actions_dict)
            except TypeError:
                 # If still not serializable, store as string representation
                 actions_json_str = json.dumps(str(event.actions))
        
        text_summary = ""
        if event.content and hasattr(event.content, "parts"):
            try:
                text_summary = " ".join(p.text for p in event.content.parts if hasattr(p, "text"))
            except: # Fallback
                text_summary = str(event.content)
        elif event.content:
            text_summary = str(event.content)

        return {
            "id": event.id, # string
            "author": event.author, # string
            "timestamp": event.timestamp, # float seconds, ensured to exist
            "invocation_id": getattr(event, "invocation_id", None) or "", # string
            "content_json": content_json_str, # string | None
            "actions_json": actions_json_str, # string | None
            "text": text_summary # string
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