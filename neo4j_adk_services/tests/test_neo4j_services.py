import unittest
import time # Used by fake_execute_write_session for timestamp
import uuid # Used by fake_execute_write_session for session_id generation if not provided

# Assuming google.adk and google.genai are available in the environment
# If not, these would need to be stubbed or mocked for pure unit testing of the service logic
try:
    from google.adk.sessions import Session
    from google.adk.events import Event, EventActions
    from google.genai import types
except ImportError:
    # Create dummy classes if ADK/GenAI types are not available
    # This allows the test structure to be valid, but tests might not be fully representative
    # In a real ADK environment, these imports should succeed.
    class Session:
        def __init__(self, app_name, user_id, id, state, events, last_update_time):
            self.app_name = app_name
            self.user_id = user_id
            self.id = id
            self.state = state
            self.events = events
            self.last_update_time = last_update_time
        
        @staticmethod
        def new_id():
            return uuid.uuid4().hex

    class Event:
        def __init__(self, author, content, actions=None, id=None, timestamp=None, invocation_id=None):
            self.id = id or uuid.uuid4().hex
            self.author = author
            self.timestamp = timestamp or time.time()
            self.invocation_id = invocation_id
            self.content = content
            self.actions = actions

        @staticmethod
        def new_id():
            return uuid.uuid4().hex

    class EventActions:
        def __init__(self, state_delta=None):
            self.state_delta = state_delta or {}

    class types:
        class Content:
            def __init__(self, parts=None):
                self.parts = parts or []
            
            def dict(self): # for json.dumps in service
                return {"parts": [p.dict() for p in self.parts]}


        class Part:
            def __init__(self, text=None):
                self.text = text
            
            def dict(self): # for json.dumps in service
                return {"text": self.text}


# Services to be tested
# We need to import neo4j to patch its driver
import neo4j
from neo4j_adk_services.src.neo4j_session_service import Neo4jSessionService, StaleSessionError # Import StaleSessionError
from neo4j_adk_services.src.neo4j_session_service import Neo4jSessionService, StaleSessionError # Import StaleSessionError
from neo4j_adk_services.src.neo4j_memory_service import Neo4jMemoryService
# Assuming SearchMemoryResponse and MemoryResult would be part of google.adk.memory
try:
    from google.adk.memory.base_memory_service import SearchMemoryResponse, MemoryResult
    from google.adk.events import Event # For dummy MemoryResult
    from google.genai.types import Content, Part # For dummy Event in MemoryResult
except ImportError:
    # This block is for when ADK types are not available.
    # The Event, Content, Part classes might already be defined above if ADK is missing.
    # Ensure they are available for the dummy MemoryResult.
    if 'Event' not in globals():
        class Event:
            def __init__(self, id, author, timestamp, content, **kwargs):
                self.id = id
                self.author = author
                self.timestamp = timestamp
                self.content = content
    if 'Content' not in globals():
        class Content:
            def __init__(self, parts=None): self.parts = parts or []
    if 'Part' not in globals():
        class Part:
            def __init__(self, text=""): self.text = text

    class MemoryResult: # Dummy for tests if not found
        def __init__(self, session_id: str, events: list[Event]): # Changed snippets to events
            self.session_id = session_id
            self.events = events # Changed snippets to events

    class SearchMemoryResponse: # Dummy for tests if not found
        def __init__(self, memories=None):
            self.memories = memories or []


class TestNeo4jServices(unittest.TestCase):
    def setUp(self):
        # Mock neo4j.GraphDatabase.driver to prevent real DB connections during service __init__
        self.original_driver = neo4j.GraphDatabase.driver
        
        class MockNeo4jRun:
            def data(self):
                return []
            def single(self):
                return None
            def consume(self):
                return None

        class MockNeo4jSession:
            def run(self, query, parameters=None, **kwargs):
                # Allow specific checks for __init__ queries if needed, e.g., constraint creation
                # For now, just return a mock run object.
                return MockNeo4jRun()
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def close(self):
                pass

        class MockNeo4jDriver:
            def session(self, database=None, fetch_size=None, impersonated_user=None, default_access_mode=None, bookmarks=None, **config):
                return MockNeo4jSession()
            def close(self):
                # print("MockNeo4jDriver closed") # For debugging
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()


        def mock_driver_fn(uri, auth, **kwargs):
            return MockNeo4jDriver()

        neo4j.GraphDatabase.driver = mock_driver_fn

        # Create service instances - now their __init__ will use the mock driver
        self.session_service = Neo4jSessionService(uri="bolt://localhost:7687", user="neo4j", password="test")
        self.memory_service = Neo4jMemoryService(uri="bolt://localhost:7687", user="neo4j", password="test", vector_dimension=128) # Added vector_dimension for init path
        
        # Monkey-patch the DB execution methods for isolation for specific test logic
        self._db = {"sessions": {}, "events": {}, "next_rels": {}} # Stores session, event, and NEXT relationship data for mock
        self._db = {"sessions": {}, "events": {}, "next_rels": {}} # Stores session, event, and NEXT relationship data for mock
        
        # Close the drivers created by the services to avoid resource warnings if tests run many times
        # In a real test with TestContainers, the container lifecycle would handle this.
        # Here, since we mock _execute_*, the driver isn't truly used by test logic after init.
        self.session_service_driver_closed = False
        self.memory_service_driver_closed = False
        
        original_session_close = self.session_service.close
        def patched_session_close():
            original_session_close()
            self.session_service_driver_closed = True
        self.session_service.close = patched_session_close

        original_memory_close = self.memory_service.close
        def patched_memory_close():
            original_memory_close()
            self.memory_service_driver_closed = True
        self.memory_service.close = patched_memory_close


        self.last_write_query_string = None # To store the last query for assertion

        def fake_execute_write_session(query, params=None):
            self.last_write_query_string = query # Capture the query
            if params is None: params = {}

            # Mock for __init__ constraint creation
            if "CREATE CONSTRAINT IF NOT EXISTS FOR (t:ToolCall) REQUIRE t.id IS UNIQUE" in query: # ToolCall constraint
                return []
            # Updated Session constraint check
            if "CREATE CONSTRAINT session_unique IF NOT EXISTS FOR (s:Session) REQUIRE (s.app_name, s.user_id, s.id) IS UNIQUE" in query:
                return []

            # Mock for create_session
            # MERGE (userNode)-[:STARTED_SESSION]->(s:Session {id: $session_id})
            if "MERGE (userNode)-[:STARTED_SESSION]->(s:Session" in query:
                sid = params["session_id"]
                app_name = params["app_name"]
                user_id = params["user_id"]
                sess_key = (app_name, user_id, sid)
                self._db["sessions"][sess_key] = {
                    "app_name": app_name,
                    "user_id": user_id,
                    "id": sid,
                    "state_json": params["state_json"],
                    "last_update_time": int(time.time() * 1000) # Store as ms, like Neo4j timestamp()
                    "last_update_time": int(time.time() * 1000) # Store as ms, like Neo4j timestamp()
                    # 'events' list will be implicitly associated by session_key in _db["events"]
                }
                # Return what the actual query returns
                return [{
                    "app_name": app_name, "user_id": user_id, "id": sid,
                    "state_json": params["state_json"], "last_update_time": self._db["sessions"][sess_key]["last_update_time"]
                    "state_json": params["state_json"], "last_update_time": self._db["sessions"][sess_key]["last_update_time"]
                }]

            # Mock for append_event
            # MATCH (s:Session ...) WHERE s.last_update_time = $client_last_update_time_ms
            # SET s.last_update_time = timestamp(), s.state_json = $new_session_state_json
            # MATCH (s:Session ...) WHERE s.last_update_time = $client_last_update_time_ms
            # SET s.last_update_time = timestamp(), s.state_json = $new_session_state_json
            # CREATE (e:Event)-[:OF_SESSION]->(s) SET e = $event_props
            # OPTIONAL MATCH (prev_event:Event)-[:OF_SESSION]->(s) ... CREATE (prev_event)-[:NEXT]->(e)
            # RETURN e.id AS event_id, s.last_update_time AS new_session_last_update_time_ms
            elif "WHERE s.last_update_time = $client_last_update_time_ms" in query and "CREATE (e:Event)-[:OF_SESSION]->(s)" in query:
            # OPTIONAL MATCH (prev_event:Event)-[:OF_SESSION]->(s) ... CREATE (prev_event)-[:NEXT]->(e)
            # RETURN e.id AS event_id, s.last_update_time AS new_session_last_update_time_ms
            elif "WHERE s.last_update_time = $client_last_update_time_ms" in query and "CREATE (e:Event)-[:OF_SESSION]->(s)" in query:
                sid = params["session_id"]
                app_name = params["app_name"]
                user_id = params["user_id"]
                sess_key = (app_name, user_id, sid)
                
                if sess_key not in self._db["sessions"]:
                    return [] # Session not found, though query implies it should exist

                # Optimistic locking check
                client_ts_ms = params["client_last_update_time_ms"]
                db_ts_ms = self._db["sessions"][sess_key]["last_update_time"]
                if client_ts_ms != db_ts_ms:
                    # This would cause the WHERE clause to fail in Neo4j, returning no rows.
                    # The service should raise StaleSessionError.
                    return []

                # Update session state and timestamp
                # Ensure new_db_timestamp_ms is greater than the current db_ts_ms
                new_db_timestamp_ms = db_ts_ms + 100 # Increment by 100ms to ensure it's different
                self._db["sessions"][sess_key]["state_json"] = params["new_session_state_json"]
                self._db["sessions"][sess_key]["last_update_time"] = new_db_timestamp_ms

                sess_key = (app_name, user_id, sid)
                
                if sess_key not in self._db["sessions"]:
                    return [] # Session not found, though query implies it should exist

                # Optimistic locking check
                client_ts_ms = params["client_last_update_time_ms"]
                db_ts_ms = self._db["sessions"][sess_key]["last_update_time"]
                if client_ts_ms != db_ts_ms:
                    # This would cause the WHERE clause to fail in Neo4j, returning no rows.
                    # The service should raise StaleSessionError.
                    return []

                # Update session state and timestamp
                # Ensure new_db_timestamp_ms is greater than the current db_ts_ms
                new_db_timestamp_ms = db_ts_ms + 100 # Increment by 100ms to ensure it's different
                self._db["sessions"][sess_key]["state_json"] = params["new_session_state_json"]
                self._db["sessions"][sess_key]["last_update_time"] = new_db_timestamp_ms

                # Store event
                event_props = params["event_props"]
                eid = event_props["id"]
                event_props = params["event_props"]
                eid = event_props["id"]
                self._db["events"][eid] = {
                    "session_key": sess_key,
                    **event_props # Store all properties from event_props (timestamp is in seconds here)
                    "session_key": sess_key,
                    **event_props # Store all properties from event_props (timestamp is in seconds here)
                }

                # Simulate NEXT relationship creation
                # Find previous event in this session
                session_events = [
                    evt for evt_id, evt in self._db["events"].items()
                    if evt.get("session_key") == sess_key and evt_id != eid
                ]
                session_events.sort(key=lambda x: x.get("timestamp", 0)) # Sort by event timestamp (seconds)
                
                if session_events:
                    prev_event_id = session_events[-1]["id"]
                    self._db["next_rels"][(prev_event_id, eid)] = True # Mark (prev_event_id)-[:NEXT]->(eid)

                # Simulate WROTE_STATE, INVOKED_TOOL if needed for more detailed tests later
                return [{"event_id": eid, "new_session_last_update_time_ms": new_db_timestamp_ms}]

            elif "MATCH (s:Session" in query and "DETACH DELETE s" in query: # For delete_session
                # Parameters from service: {"app": app_name, "user": user_id, "sid": session_id}
                app_name_param = params.get("app")
                user_id_param = params.get("user")
                session_id_param = params.get("sid")
                
                sess_key = (app_name_param, user_id_param, session_id_param)
                # Remove all events for this session
                for eid, evt_data in list(self._db["events"].items()):
                    if evt_data["session_key"] == sess_key:
                        self._db["events"].pop(eid)
                self._db["sessions"].pop(sess_key, None)
                return []
            return []

        def fake_execute_read_session(query, params=None):
            if params is None: params = {}
            # MATCH (s:Session {app_name: $app_name, user_id: $user_id, id: $session_id}) OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:Event)
            # The new query uses (:Event)-[:OF_SESSION]->(s)
            if "RETURN s, collect(e) AS events" in query: # For get_session
                sess_key = (params["app_name"], params["user_id"], params["session_id"])
                session_node_data = self._db["sessions"].get(sess_key)
                if not session_node_data:
                    return [] # Session not found

                # Collect events for this session
                # Event nodes are now linked via (:Event)-[:OF_SESSION]->(s)
                # Our mock stores events with a "session_key"
                collected_events_data = []
                for eid, event_data_in_db in self._db["events"].items():
                    if event_data_in_db.get("session_key") == sess_key:
                        # Simulate the structure Neo4j driver might return for a node
                        # The service's get_session expects dict(e_node)
                        # event_data_in_db already contains the necessary props from _prepare_event_properties
                        collected_events_data.append(event_data_in_db)
                
                # Sort events by timestamp as the actual query does (event timestamps are in seconds)
                # Sort events by timestamp as the actual query does (event timestamps are in seconds)
                collected_events_data.sort(key=lambda e: e.get("timestamp", 0))
                
                # The actual query returns `s` (the session node) and `collect(e) AS events`
                # `s_node` in get_session becomes `record['s']` (last_update_time is in ms)
                # `events_list` becomes `record['events']` (event timestamps are in seconds)
                # `s_node` in get_session becomes `record['s']` (last_update_time is in ms)
                # `events_list` becomes `record['events']` (event timestamps are in seconds)
                return [{"s": session_node_data, "events": collected_events_data}]

            # For list_sessions: MATCH (s:Session {app_name: $app_name, user_id: $user_id}) RETURN s.id AS session_id, s.last_update_time AS last_update
            elif "RETURN s.id AS session_id, s.last_update_time AS last_update" in query:
                # list_sessions query
                app, user = params["app_name"], params["user_id"]
                result = []
                for (a, u, sid), sess_data in self._db["sessions"].items():
                    if a == app and u == user:
                        # last_update_time in mock DB is ms, service expects it to be converted to Session obj (seconds)
                        # but list_sessions directly uses the returned value.
                        # For consistency, let's assume the mock returns it as the service would expect for Session obj construction.
                        # However, the service's list_sessions method creates Session objects where last_update_time is float seconds.
                        # So, the mock should return it in ms as it's stored in the DB.
                        result.append({"session_id": sid, "last_update": sess_data["last_update_time"]}) # ms
                        # last_update_time in mock DB is ms, service expects it to be converted to Session obj (seconds)
                        # but list_sessions directly uses the returned value.
                        # For consistency, let's assume the mock returns it as the service would expect for Session obj construction.
                        # However, the service's list_sessions method creates Session objects where last_update_time is float seconds.
                        # So, the mock should return it in ms as it's stored in the DB.
                        result.append({"session_id": sid, "last_update": sess_data["last_update_time"]}) # ms
                return result
            
            # For NEXT relationship check in tests
            # MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c
            elif "MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c" in query:
                e1_id = params.get("e1")
                e3_id = params.get("e3")
                # Check path: e1 -> e2 -> e3
                # Find e2 such that (e1)-[:NEXT]->(e2)
                e2_id = None
                for (start_node, end_node) in self._db["next_rels"].keys():
                    if start_node == e1_id:
                        e2_id = end_node
                        break
                if not e2_id: return []

                # Check if (e2)-[:NEXT]->(e3)
                if self._db["next_rels"].get((e2_id, e3_id)):
                    # Return the 'c' node (e3)
                    if e3_id in self._db["events"]:
                         # Simulate returning the node properties as Neo4j would
                        return [{"c": self._db["events"][e3_id]}]
                return []

            
            # For NEXT relationship check in tests
            # MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c
            elif "MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c" in query:
                e1_id = params.get("e1")
                e3_id = params.get("e3")
                # Check path: e1 -> e2 -> e3
                # Find e2 such that (e1)-[:NEXT]->(e2)
                e2_id = None
                for (start_node, end_node) in self._db["next_rels"].keys():
                    if start_node == e1_id:
                        e2_id = end_node
                        break
                if not e2_id: return []

                # Check if (e2)-[:NEXT]->(e3)
                if self._db["next_rels"].get((e2_id, e3_id)):
                    # Return the 'c' node (e3)
                    if e3_id in self._db["events"]:
                         # Simulate returning the node properties as Neo4j would
                        return [{"c": self._db["events"][e3_id]}]
                return []

            return []

        self.session_service._execute_write = fake_execute_write_session
        self.session_service._execute_read = fake_execute_read_session

        # Memory service fake DB interactions
        # For add_session_to_memory, it calls _execute_write. We need a mock for that.
        # The MemoryService's _execute_write is used to set :Memory label and embedding.
        def fake_execute_write_memory(query, params=None):
            if params is None: params = {}
            # Mock for __init__ index creation
            if "CALL db.index.fulltext.createNodeIndex('MemoryTextIndex', ['Memory']" in query: # Original label was Memory
                return []
            if "CALL db.index.fulltext.createNodeIndex('MemoryTextIndex', ['MemoryChunk']" in query: # New label is MemoryChunk
                return []
            if "CALL db.index.vector.createNodeIndex('MemoryVectorIndex', 'Memory'" in query: # Original label
                return []
            if "CALL db.index.vector.createNodeIndex('MemoryVectorIndex', 'MemoryChunk'" in query: # New label
                return []

            # Mock for add_session_to_memory
            # UNWIND $events AS chunk_data MERGE (mc:MemoryChunk {eid: chunk_data.eid}) SET mc += chunk_data
            if "UNWIND $events AS chunk_data" in query and "MERGE (mc:MemoryChunk {eid: chunk_data.eid})" in query:
                chunks_to_add = params.get("events", []) # Parameter name is 'events' in service
                added_count = 0
                if "memory_chunks" not in self._db:
                    self._db["memory_chunks"] = {}
                
                for chunk_data in chunks_to_add:
                    eid = chunk_data["eid"]
                    # Store the chunk data. Using eid as key for simplicity in mock.
                    self._db["memory_chunks"][eid] = {
                        "eid": eid,
                        "text": chunk_data["text"],
                        "author": chunk_data["author"],
                        "ts": chunk_data["ts"],
                        "app_name": params.get("app"), # app_name from parameters
                        "user_id": params.get("user"), # user_id from parameters
                        "session_id": params.get("sid"), # session_id from parameters
                        "embedding": chunk_data.get("embedding")
                    }
                    added_count += 1
                return [{"added": added_count}]
            return []
        
        self.memory_service._execute_write = fake_execute_write_memory

        def fake_execute_read_memory(query, params=None):
            if params is None: params = {}
            # Simulate fulltext search
            # Simulate fulltext search on MemoryChunk nodes
            if "CALL db.index.fulltext.queryNodes('MemoryTextIndex'" in query:
                q_text = params["query"].lower()
                app_name_param = params["app_name"]
                user_id_param = params["user_id"]
                mock_results = []
                for eid, chunk_data in self._db.get("memory_chunks", {}).items():
                    text_match = q_text in chunk_data.get("text", "").lower()
                    app_match = chunk_data.get("app_name") == app_name_param
                    user_match = chunk_data.get("user_id") == user_id_param
                    if text_match and app_match and user_match:
                        # session_id is now expected to be directly on the chunk_data from the mock DB
                        mock_results.append({
                            "session_id": chunk_data.get("session_id"), # Use directly stored session_id
                            "event_id": chunk_data.get("eid"),
                            "text": chunk_data.get("text"),
                            "author": chunk_data.get("author"),
                            "ts": chunk_data.get("ts"),
                            "score": 1.0  # dummy score
                        })
                return mock_results
            
            # Simulate vector search on MemoryChunk nodes
            if "CALL db.index.vector.queryNodes('MemoryVectorIndex'" in query:
                # This would require a mock embedding function and vector comparison logic
                return [] # Return no vector results for simplicity in this mock
            return []
        
        self.memory_service._execute_read = fake_execute_read_memory

    def tearDown(self):
        # Restore the original neo4j.GraphDatabase.driver
        neo4j.GraphDatabase.driver = self.original_driver

        # Ensure drivers created by services (even if they are mocks from the initial patch) are "closed"
        # The patched close methods in setUp handle the self.xxx_driver_closed flags
        if hasattr(self.session_service, 'close'):
            self.session_service.close()
        if hasattr(self.memory_service, 'close'):
            self.memory_service.close()


    def test_create_and_get_session(self):
        # Create a session
        sess = self.session_service.create_session(app_name="test_app", user_id="user123", state={"foo": "bar"})
        self.assertIsInstance(sess, Session)
        self.assertEqual(sess.app_name, "test_app")
        self.assertEqual(sess.user_id, "user123")
        # The initial state should be as provided
        self.assertEqual(sess.state.get("foo"), "bar")
        # last_update_time should be a float (seconds)
        self.assertIsInstance(sess.last_update_time, float)

        # last_update_time should be a float (seconds)
        self.assertIsInstance(sess.last_update_time, float)

        # Retrieving the session should return the same data
        fetched = self.session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, sess.id)
        self.assertEqual(fetched.state, {"foo": "bar"}) # State is JSON dumped and loaded, should be equal
        self.assertEqual(len(fetched.events), 0)  # no events yet
        self.assertIsInstance(fetched.last_update_time, float)
        self.assertAlmostEqual(fetched.last_update_time, sess.last_update_time, places=2)

        self.assertIsInstance(fetched.last_update_time, float)
        self.assertAlmostEqual(fetched.last_update_time, sess.last_update_time, places=2)


    def test_append_event_state_update(self):
        sess = self.session_service.create_session(app_name="test_app", user_id="user123", state={"count": 1})
        original_last_update_time = sess.last_update_time

        original_last_update_time = sess.last_update_time

        # Create a dummy Event with state_delta action
        evt_actions = EventActions(state_delta={"count": 2, "temp:note": "temp", "new_key": "value", "remove_this": None})
        evt_content = types.Content(parts=[types.Part(text="Hello world")])
        evt = Event(author="user", content=evt_content, actions=evt_actions)
        
        # Append the event
        returned_event = self.session_service.append_event(sess, evt)
        self.assertEqual(returned_event, evt) # append_event should return the event passed in (possibly modified with id/timestamp)

        # The session object should be updated in memory
        self.assertEqual(sess.state.get("count"), 2)          # updated by state_delta
        self.assertNotIn("temp:note", sess.state)             # temp key not persisted by service logic
        self.assertEqual(sess.state.get("new_key"), "value")  # new key added
        self.assertNotIn("remove_this", sess.state)           # None value should remove key
        self.assertEqual(len(sess.events), 1)
        self.assertEqual(sess.events[0], evt)

        # The event should have an id assigned and be stored in our mock _db
        self.assertIsNotNone(evt.id) # ID should be assigned by append_event if not present
        evt_stored_in_mock = self._db["events"].get(evt.id)
        self.assertIsNotNone(evt_stored_in_mock)
        self.assertEqual(evt_stored_in_mock["text"], "Hello world") # Check text field was generated

        # Session last_update_time should be updated by append_event from the DB mock
        self.assertNotAlmostEqual(sess.last_update_time, original_last_update_time, places=5)
        self.assertTrue(sess.last_update_time > original_last_update_time)
        # Session last_update_time should be updated by append_event from the DB mock
        self.assertNotAlmostEqual(sess.last_update_time, original_last_update_time, places=5)
        self.assertTrue(sess.last_update_time > original_last_update_time)
        
        # Now retrieve session from service and check state and events
        fetched = self.session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.state.get("count"), 2)
        self.assertIn("new_key", fetched.state)
        self.assertNotIn("remove_this", fetched.state)
        self.assertNotIn("temp:note", fetched.state) # Temp keys should not be in fetched state either

        # The fetched session should have one event with matching content
        self.assertEqual(len(fetched.events), 1)
        fetched_event = fetched.events[0]
        self.assertEqual(fetched_event.id, evt.id)
        self.assertEqual(fetched_event.author, evt.author)
        self.assertTrue(fetched_event.content and fetched_event.content.parts[0].text == "Hello world")
        # Check actions reconstruction (simplified, as EventActions might not be fully reconstructed by default mock)
        # In the real service, EventActions are reconstructed if actions_json is present.
        # Our mock for get_session populates actions_json, so the service should try to parse it.
        # For this test, we ensure the original event's actions are used for comparison if needed.
        if fetched_event.actions: # If actions were reconstructed
            self.assertEqual(fetched_event.actions.state_delta, evt_actions.state_delta)


    def test_search_memory_fulltext(self):
        # Create a session and append events
        sess = self.session_service.create_session(app_name="test_app", user_id="user123")
        evt1_content = types.Content(parts=[types.Part(text="Paris is the capital of France.")])
        evt1 = Event(author="user", content=evt1_content, actions=EventActions())
        self.session_service.append_event(sess, evt1)
        
        evt2_content = types.Content(parts=[types.Part(text="Sure, noted.")])
        evt2 = Event(author="agent", content=evt2_content, actions=EventActions())
        self.session_service.append_event(sess, evt2)
        
        # Ingest session to memory
        self.memory_service.add_session_to_memory(sess)

        # Verify that memory chunks are created in _db["memory_chunks"]
        # The key for memory_chunks in the mock is the event ID (eid)
        self.assertIn(evt1.id, self._db.get("memory_chunks", {}), "Event 1 should have a corresponding memory chunk.")
        self.assertEqual(self._db["memory_chunks"][evt1.id].get("text"), "Paris is the capital of France.")
        self.assertIn(evt2.id, self._db.get("memory_chunks", {}), "Event 2 should have a corresponding memory chunk.")
        
        # Search for a keyword present in evt1 content
        response = self.memory_service.search_memory(app_name="test_app", user_id="user123", query="capital of France")
        
        self.assertIsInstance(response, SearchMemoryResponse, "Search memory should return a SearchMemoryResponse object.")
        
        memories_list = response.memories
        self.assertIsNotNone(memories_list, "Response memories list should not be None.")

        # Check if any MemoryResult was found. The mock might return an empty list if no match.
        # If a match is expected, assertGreaterEqual(len(memories_list), 1)
        # For this specific query "capital of France", we expect a match.
        self.assertGreaterEqual(len(memories_list), 1, "Should find at least one matching MemoryResult.")
        
        found_event_text = False
        for mem_item in memories_list: # mem_item is now expected to be a MemoryResult object
            self.assertIsInstance(mem_item, MemoryResult, "Each item in memories should be a MemoryResult object.")
            if mem_item.session_id == sess.id:
                self.assertTrue(hasattr(mem_item, 'events'), "MemoryResult should have an 'events' attribute.")
                for event_in_memory in mem_item.events:
                    self.assertIsInstance(event_in_memory, Event, "Each event in MemoryResult should be an Event object.")
                    # Assuming the reconstructed event has content with parts
                    if event_in_memory.content and hasattr(event_in_memory.content, 'parts'):
                        for part in event_in_memory.content.parts:
                            if hasattr(part, 'text') and "Paris is the capital of France." in part.text:
                                found_event_text = True
                                break
                    if found_event_text:
                        break
            if found_event_text:
                break
        
        self.assertTrue(found_event_text, "Memory search should retrieve an event with the expected text.")

    def test_list_sessions(self):
        self.session_service.create_session(app_name="app1", user_id="user1", state={"data": "session1"})
        self.session_service.create_session(app_name="app1", user_id="user1", state={"data": "session2"})
        self.session_service.create_session(app_name="app1", user_id="user2", state={"data": "session3"}) # Different user

        listed_sessions_response = self.session_service.list_sessions(app_name="app1", user_id="user1")
        self.assertEqual(len(listed_sessions_response.sessions), 2)
        for s in listed_sessions_response.sessions:
            self.assertEqual(s.app_name, "app1")
            self.assertEqual(s.user_id, "user1")

    def test_delete_session(self):
        sess_to_delete = self.session_service.create_session(app_name="app_del", user_id="user_del")
        self.session_service.append_event(sess_to_delete, Event(author="test", content=types.Content(parts=[types.Part(text="event to delete")])))
        
        # Check it exists
        self.assertIsNotNone(self.session_service.get_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id))
        self.assertIn(sess_to_delete.events[0].id, self._db["events"]) # Check event in mock db

        # Delete
        self.session_service.delete_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)

        # Verify it's gone
        self.assertIsNone(self.session_service.get_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id))
        self.assertNotIn(sess_to_delete.events[0].id, self._db["events"]) # Check event also removed from mock db
        self.assertNotIn(("app_del", "user_del", sess_to_delete.id), self._db["sessions"])


def test_wrote_state_edge_creation(self):
    """Test that WROTE_STATE edge is part of the Cypher query when state changes."""
    sess = self.session_service.create_session(app_name="test_app_state", user_id="user_state")
    
    # Ensure the mock is in place for _execute_write
    # This is to capture the query string from append_event
    original_execute_write = self.session_service._execute_write
    
    captured_query_in_test = None
    def capture_query_for_append(query, params=None):
        nonlocal captured_query_in_test
        captured_query_in_test = query
        # Call the original fake_execute_write_session to maintain mock DB state
        return self.session_service._original_fake_execute_write(query, params)

    # Temporarily replace _execute_write with our capturer
    self.session_service._original_fake_execute_write = self.session_service._execute_write
    self.session_service._execute_write = capture_query_for_append

    evt_actions = EventActions(state_delta={'flag': 'on'})
    # Use dummy Content and Part if not available from google.genai.types
    content_parts = [types.Part(text='toggle') if 'types' in globals() and hasattr(types, 'Part') else {"text": "toggle"}]
    content_obj = types.Content(parts=content_parts) if 'types' in globals() and hasattr(types, 'Content') else {"parts": content_parts}
    
    evt = Event(author='user', content=content_obj, actions=evt_actions)
    self.session_service.append_event(sess, evt)

    # Restore original mock
    self.session_service._execute_write = self.session_service._original_fake_execute_write
    del self.session_service._original_fake_execute_write

    self.assertIsNotNone(captured_query_in_test, "Cypher query should have been captured.")
    self.assertIn('WROTE_STATE', captured_query_in_test,
                    "The Cypher query for append_event should include 'WROTE_STATE'.")
    self.assertIn('WHERE s.last_update_time = $client_last_update_time_ms', captured_query_in_test, "Optimistic locking check missing.")
    self.assertIn('CREATE (prev_event)-[:NEXT]->(e)', captured_query_in_test, "NEXT relationship creation missing.")


    def test_optimistic_locking_stale_session(self):
        sess = self.session_service.create_session(app_name="test_app_stale", user_id="user_stale")
        
        # Simulate that the session was updated by another process in the DB
        # Our mock DB stores last_update_time in ms
        mock_db_session_key = (sess.app_name, sess.user_id, sess.id)
        self._db["sessions"][mock_db_session_key]["last_update_time"] += 1000 # Advance DB time by 1s (in ms)

        evt_content = types.Content(parts=[types.Part(text="Attempting update on stale session")])
        evt = Event(author="user", content=evt_content, actions=EventActions(state_delta={"key": "value"}))
        
        with self.assertRaises(StaleSessionError):
            self.session_service.append_event(sess, evt)

    def test_event_chaining_next_relationship(self):
        sess = self.session_service.create_session(app_name="test_app_chain", user_id="user_chain")
        
        evt1_content = types.Content(parts=[types.Part(text="Event 1")])
        evt1 = Event(author="user", content=evt1_content, id="evt1_chain") # Provide ID for easier testing
        self.session_service.append_event(sess, evt1)
        
        # Ensure session's last_update_time is updated in the Python object for the next append
        # The mock DB's fake_execute_write_session for append_event returns new_session_last_update_time_ms
        # The service updates sess.last_update_time with this.

        evt2_content = types.Content(parts=[types.Part(text="Event 2")])
        evt2 = Event(author="user", content=evt2_content, id="evt2_chain")
        self.session_service.append_event(sess, evt2)

        evt3_content = types.Content(parts=[types.Part(text="Event 3")])
        evt3 = Event(author="user", content=evt3_content, id="evt3_chain")
        self.session_service.append_event(sess, evt3)

        # Verify NEXT relationships in the mock DB
        self.assertTrue(self._db["next_rels"].get(("evt1_chain", "evt2_chain")))
        self.assertTrue(self._db["next_rels"].get(("evt2_chain", "evt3_chain")))
        self.assertFalse(self._db["next_rels"].get(("evt1_chain", "evt3_chain"))) # Should not exist directly

        # Test the path query as per issue description (using fake_execute_read_session)
        # This requires fake_execute_read_session to handle this specific query pattern.
        path_query_params = {"e1": "evt1_chain", "e3": "evt3_chain"}
        # The query is: "MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c"
        # Our fake_execute_read_session is already set up to handle this.
        result = self.session_service._execute_read(
            "MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c",
            path_query_params
        )
        self.assertTrue(len(result) > 0, "Path query should find e3 via e1->e2->e3.")
        self.assertEqual(result[0]["c"]["id"], "evt3_chain", "The returned node 'c' should be e3.")


if __name__ == '__main__':
    unittest.main()