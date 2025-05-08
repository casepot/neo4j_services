import unittest
import time # Used by fake_execute_write_session for timestamp
import uuid # Used by fake_execute_write_session for session_id generation if not provided
import json # For checking JSON serialization in tests if needed

# Assuming google.adk and google.genai are available in the environment
# If not, these would need to be stubbed or mocked for pure unit testing of the service logic
# Import necessary ADK classes directly
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import ListSessionsResponse, ListEventsResponse
from google.adk.events import Event, EventActions
# ToolCall is no longer in google.adk.events, use FunctionCall from google.genai.types
from google.genai import types
from google.genai.types import FunctionCall as ToolCall # Use alias for compatibility
from google.adk.memory.base_memory_service import SearchMemoryResponse, MemoryResult


# Services to be tested
# We need to import neo4j to patch its driver
import neo4j
from neo4j_adk_services.src.neo4j_session_service import Neo4jSessionService, StaleSessionError # Import StaleSessionError
from neo4j_adk_services.src.neo4j_memory_service import Neo4jMemoryService
# No need for dummy MemoryResult/SearchMemoryResponse definitions anymore


class TestNeo4jServices(unittest.TestCase):
    def setUp(self): # Standard setUp signature
        # Mock neo4j.GraphDatabase.driver to prevent real DB connections during service __init__
        self.original_driver = neo4j.GraphDatabase.driver # Use self directly
        self.simulated_existing_indexes = set() # To track indexes for simulating "already exists"
        
        # Capture the 'self' of TestNeo4jServices for use in nested functions/classes
        test_case_instance_outer = self

        class MockNeo4jRun:
            def data(self):
                return []
            def single(self):
                return None
            def consume(self):
                return None

        class MockNeo4jSession:
            def __init__(self, test_case_ref): # test_case_ref is the TestNeo4jServices instance
                self.test_case_ref = test_case_ref

            def run(self, query, parameters=None, **kwargs):
                # Simulate behavior for DDL statements in __init__
                if "CREATE CONSTRAINT" in query and "IF NOT EXISTS" in query:
                    # Simulate constraint creation for AppState and UserState as well
                    if "FOR (a:AppState)" in query and 'app_state_unique' in self.test_case_ref.simulated_existing_indexes:
                         raise neo4j.exceptions.ClientError("An equivalent constraint already exists: app_state_unique")
                    if "FOR (a:AppState)" in query:
                        self.test_case_ref.simulated_existing_indexes.add('app_state_unique')
                    
                    if "FOR (u:UserState)" in query and 'user_state_unique' in self.test_case_ref.simulated_existing_indexes:
                        raise neo4j.exceptions.ClientError("An equivalent constraint already exists: user_state_unique")
                    if "FOR (u:UserState)" in query:
                        self.test_case_ref.simulated_existing_indexes.add('user_state_unique')

                    return MockNeo4jRun()
                if "CALL db.index.fulltext.createNodeIndex('MemoryTextIndex', ['MemoryChunk']" in query:
                    if 'MemoryTextIndex' in self.test_case_ref.simulated_existing_indexes:
                        raise neo4j.exceptions.ClientError("An equivalent index already exists: MemoryTextIndex")
                    self.test_case_ref.simulated_existing_indexes.add('MemoryTextIndex')
                    return MockNeo4jRun()
                if "CALL db.index.vector.createNodeIndex('MemoryVectorIndex', 'MemoryChunk'" in query:
                    if 'MemoryVectorIndex' in self.test_case_ref.simulated_existing_indexes:
                        raise neo4j.exceptions.ClientError("Error: An equivalent index already exists: MemoryVectorIndex")
                    self.test_case_ref.simulated_existing_indexes.add('MemoryVectorIndex')
                    return MockNeo4jRun()
                return MockNeo4jRun() # Default for other init queries

            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def close(self):
                pass

        class MockNeo4jDriver:
            def __init__(self, test_case_ref): # test_case_ref is the TestNeo4jServices instance
                self.test_case_ref = test_case_ref

            def session(self, database=None, fetch_size=None, impersonated_user=None, default_access_mode=None, bookmarks=None, **config):
                return MockNeo4jSession(self.test_case_ref) # Pass it to the session
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()

        # mock_driver_fn now correctly uses the captured test_case_instance_outer (which is 'self' from setUp)
        def mock_driver_fn(uri, auth, **kwargs):
            return MockNeo4jDriver(test_case_instance_outer) 

        neo4j.GraphDatabase.driver = mock_driver_fn

        # Create service instances - now their __init__ will use the mock driver
        self.session_service = Neo4jSessionService(uri="bolt://localhost:7687", user="neo4j", password="test")
        self.memory_service = Neo4jMemoryService(uri="bolt://localhost:7687", user="neo4j", password="test", vector_dimension=128)
        
        try: # Test re-initialization (idempotency check)
            Neo4jSessionService(uri="bolt://localhost:7687", user="neo4j", password="test")
            Neo4jMemoryService(uri="bolt://localhost:7687", user="neo4j", password="test", vector_dimension=128)
        except neo4j.exceptions.ClientError as e:
            if "already exists" not in str(e).lower() and "an equivalent index already exists" not in str(e).lower():
                self.fail(f"Service re-initialization failed with unexpected ClientError: {e}")
        except Exception as e:
            self.fail(f"Service re-initialization failed unexpectedly: {e}")

        # Mock database dictionary, attached to the test case instance
        self._db = {
            "sessions": {}, "events": {}, "next_rels": {},
            "memory_chunks": {}, "tool_calls": {},
            "app_states": {}, "user_states": {} # For P7
        }
        
        # Patch close methods to track closure
        self.session_service_driver_closed = False
        self.memory_service_driver_closed = False
        
        original_session_close = self.session_service.close
        def patched_session_close():
            nonlocal original_session_close # Ensure closure captures the right variable
            original_session_close()
            test_case_instance_outer.session_service_driver_closed = True # Use captured outer self
        self.session_service.close = patched_session_close

        original_memory_close = self.memory_service.close
        def patched_memory_close():
            nonlocal original_memory_close
            original_memory_close()
            test_case_instance_outer.memory_service_driver_closed = True # Use captured outer self
        self.memory_service.close = patched_memory_close

        # Store last query for potential assertions
        self.last_write_query_string = None

        # Define mock execute functions using the captured outer self (test_case_instance_outer)
        def fake_execute_write_session(query, params=None):
            test_case_instance_outer.last_write_query_string = query
            _db = test_case_instance_outer._db # Local alias for convenience
            if params is None: params = {}

            # Ignore DDL during tests (already handled by mock session in setUp)
            if "CREATE CONSTRAINT" in query: return []
            
            # Mock for create_session
            if "MERGE (userNode)-[:STARTED_SESSION]->(s:Session" in query:
                sid, app_name, user_id = params["session_id"], params["app_name"], params["user_id"]
                sess_key = (app_name, user_id, sid)
                current_time_ms = int(time.time() * 1000)
                
                # P7: state_json in params already includes merged shadow states from fake_execute_read_session
                _db["sessions"][sess_key] = {
                    "app_name": app_name, "user_id": user_id, "id": sid,
                    "state_json": params["state_json"], "last_update_time": current_time_ms
                }
                return [{"app_name": app_name, "user_id": user_id, "id": sid,
                         "state_json": params["state_json"], "last_update_time": current_time_ms}]
            
            # Mock for append_event
            # This mock needs to handle P7 shadow state updates now
            elif "FOREACH (ignoreMe IN CASE WHEN size(keys($app_state_delta)) > 0 THEN [1] ELSE [] END |" in query or \
                 "WHERE s.last_update_time = $client_last_update_time_ms" in query and "CREATE (e:Event)-[:OF_SESSION]->(s)" in query:
                
                app_name = params["app_name"]
                user_id = params["user_id"]
                
                # P7: Simulate AppState update
                app_state_delta = params.get("app_state_delta", {})
                if app_state_delta:
                    app_key = app_name
                    current_app_state_json = _db["app_states"].get(app_key, {}).get("state_json", "{}")
                    current_app_state = json.loads(current_app_state_json)
                    # Simulate apoc.map.merge
                    for k, v in app_state_delta.items():
                        if v is None: current_app_state.pop(k, None)
                        else: current_app_state[k] = v
                    _db["app_states"][app_key] = {
                        "state_json": json.dumps(current_app_state),
                        "version": int(time.time() * 1000)
                    }

                # P7: Simulate UserState update
                user_state_delta = params.get("user_state_delta", {})
                if user_state_delta:
                    user_key = (app_name, user_id)
                    current_user_state_json = _db["user_states"].get(user_key, {}).get("state_json", "{}")
                    current_user_state = json.loads(current_user_state_json)
                    for k, v in user_state_delta.items():
                        if v is None: current_user_state.pop(k, None)
                        else: current_user_state[k] = v
                    _db["user_states"][user_key] = {
                        "state_json": json.dumps(current_user_state),
                        "version": int(time.time() * 1000)
                    }

                # Continue with existing append_event mock logic for Session and Event
                sid = params["session_id"]
                sess_key = (app_name, user_id, sid)
                if sess_key not in _db["sessions"]: return []
                
                client_ts_ms = params["client_last_update_time_ms"]
                db_ts_ms = _db["sessions"][sess_key]["last_update_time"]
                if client_ts_ms != db_ts_ms: return [] # Simulate optimistic lock failure

                new_db_timestamp_ms = db_ts_ms + 100
                _db["sessions"][sess_key]["state_json"] = params["new_session_state_json"]
                _db["sessions"][sess_key]["last_update_time"] = new_db_timestamp_ms
                
                event_props = params["event_props"]
                eid = event_props["id"]
                _db["events"][eid] = {"session_key": sess_key, **event_props}
                
                # Simulate NEXT relationship
                session_events = [evt for evt_id, evt in _db["events"].items() if evt.get("session_key") == sess_key and evt_id != eid]
                session_events.sort(key=lambda x: x.get("timestamp", 0))
                if session_events:
                    prev_event_id = session_events[-1]["id"]
                    _db["next_rels"][(prev_event_id, eid)] = True

                # Simulate ToolCall creation
                tool_calls_data = params.get("tool_calls_data", [])
                for tc_data in tool_calls_data:
                    _db["tool_calls"][tc_data["id"]] = tc_data # Store tool call data

                return [{"event_id": eid, "new_session_last_update_time_ms": new_db_timestamp_ms}]
            
            # Mock for delete_session
            elif "MATCH (s:Session" in query and "DETACH DELETE s" in query:
                app_name, user_id, sid = params.get("app"), params.get("user"), params.get("sid")
                sess_key = (app_name, user_id, sid)
                # Remove associated events and tool calls (simplified mock - assumes no complex relations from events/tools)
                for eid, evt_data in list(_db["events"].items()):
                    if evt_data["session_key"] == sess_key: 
                        _db["events"].pop(eid)
                        # Also remove related NEXT rels
                        _db["next_rels"] = {k: v for k, v in _db["next_rels"].items() if k[0] != eid and k[1] != eid}
                        # Mock doesn't track INVOKED_TOOL rels explicitly for deletion here, assumes DETACH handles it
                _db["sessions"].pop(sess_key, None)
                return []
                
            return [] # Default empty result

        def fake_execute_read_session(query, params=None):
            _db = test_case_instance_outer._db # Local alias
            if params is None: params = {}

            # For get_session: Check for key elements
            if "MATCH (s:Session {" in query and \
               "OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:Event)" in query and \
               "RETURN s, collect(e) AS events" in query: # This is get_session
                sess_key = (params["app_name"], params["user_id"], params["session_id"])
                session_node_data = _db["sessions"].get(sess_key)
                if not session_node_data: return [] # Session not found in mock
                
                # Collect events, ensuring they are copies to avoid modifying mock DB directly if needed
                collected_events_data = [dict(data) for eid, data in _db["events"].items() if data.get("session_key") == sess_key]
                collected_events_data.sort(key=lambda e: e.get("timestamp", 0))
                # Return a copy of the session data as well
                return [{"s": dict(session_node_data), "events": collected_events_data}]

            # P7: Mock for create_session's AppState read
            elif "OPTIONAL MATCH (as:AppState {app_name: $app_name})" in query:
                app_name_param = params["app_name"]
                app_state_data = _db["app_states"].get(app_name_param)
                if app_state_data:
                    return [{"app_state_json": app_state_data.get("state_json"), "app_state_version": app_state_data.get("version")}]
                return [{"app_state_json": None, "app_state_version": None}]

            # P7: Mock for create_session's UserState read
            elif "OPTIONAL MATCH (us:UserState {app_name: $app_name, user_id: $user_id})" in query:
                user_key_param = (params["app_name"], params["user_id"])
                user_state_data = _db["user_states"].get(user_key_param)
                if user_state_data:
                    return [{"user_state_json": user_state_data.get("state_json"), "user_state_version": user_state_data.get("version")}]
                return [{"user_state_json": None, "user_state_version": None}]

            # For list_sessions:
            elif "RETURN s.id AS session_id, s.last_update_time AS last_update" in query:
                app, user = params["app_name"], params["user_id"]
                return [{"session_id": sid, "last_update": data["last_update_time"]} 
                        for (a, u, sid), data in _db["sessions"].items() if a == app and u == user]

            # For list_events:
            elif "MATCH (s:Session" in query and "ORDER BY e.timestamp" in query and \
                 "RETURN collect(e) AS events" in query and "RETURN s," not in query:
                sess_key = (params["app_name"], params["user_id"], params["session_id"])
                if sess_key not in _db["sessions"]: return [{"events": []}] 
                collected_events_data = [dict(data) for eid, data in _db["events"].items() if data.get("session_key") == sess_key]
                collected_events_data.sort(key=lambda ev_node: ev_node.get("timestamp", 0))
                return [{"events": collected_events_data}]
            
            # For NEXT relationship check in tests
            elif "MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c" in query:
                e1_id, e3_id = params.get("e1"), params.get("e3")
                e2_id = next((end_node for start_node, end_node in _db["next_rels"].keys() if start_node == e1_id), None)
                if not e2_id or not _db["next_rels"].get((e2_id, e3_id)) or e3_id not in _db["events"]: return []
                # Return a copy of the event data
                return [{"c": dict(_db["events"][e3_id])}]
                
            return [] # Default empty result

        # Assign mocks to the service instance
        self.session_service._execute_write = fake_execute_write_session
        self.session_service._execute_read = fake_execute_read_session

        # --- Memory Service Mocks ---
        def fake_execute_write_memory(query, params=None):
            _db = test_case_instance_outer._db
            if params is None: params = {}
            # Ignore DDL
            if "CALL db.index.fulltext.createNodeIndex" in query: return []
            if "CALL db.index.vector.createNodeIndex" in query: return []
            
            # Mock for add_session_to_memory
            if "UNWIND $events AS chunk_data" in query and "MERGE (mc:MemoryChunk {eid: chunk_data.eid})" in query:
                chunks_to_add, added_count = params.get("events", []), 0
                if "memory_chunks" not in _db: _db["memory_chunks"] = {}
                for chunk_data in chunks_to_add:
                    eid = chunk_data["eid"]
                    _db["memory_chunks"][eid] = {
                        "eid": eid, "text": chunk_data["text"], "author": chunk_data["author"], "ts": chunk_data["ts"],
                        "app_name": params.get("app"), "user_id": params.get("user"), "session_id": params.get("sid"),
                        "embedding": chunk_data.get("embedding")}
                    added_count += 1
                return [{"added": added_count}]
            return []
        
        def fake_execute_read_memory(query, params=None):
            _db = test_case_instance_outer._db
            if params is None: params = {}
            # Mock for search_memory (full-text)
            if "CALL db.index.fulltext.queryNodes('MemoryTextIndex'" in query:
                q_text, app_param, user_param = params["query"].lower(), params["app_name"], params["user_id"]
                results = []
                for eid, chunk in _db.get("memory_chunks", {}).items():
                    if q_text in chunk.get("text", "").lower() and \
                       chunk.get("app_name") == app_param and \
                       chunk.get("user_id") == user_param:
                        # Return data needed by search_memory to reconstruct MemoryResult
                        results.append({"session_id": chunk.get("session_id"), "event_id": chunk.get("eid"), 
                                        "text": chunk.get("text"), "author": chunk.get("author"), 
                                        "ts": chunk.get("ts"), "score": 1.0})
                return results
            # Mock for search_memory (vector) - returning empty for simplicity
            if "CALL db.index.vector.queryNodes('MemoryVectorIndex'" in query: 
                return []
            return []
        
        self.memory_service._execute_write = fake_execute_write_memory
        self.memory_service._execute_read = fake_execute_read_memory

    def tearDown(self):
        # Restore the original neo4j.GraphDatabase.driver
        neo4j.GraphDatabase.driver = self.original_driver

        # Ensure drivers created by services are "closed"
        if hasattr(self.session_service, 'close'):
            self.session_service.close()
        if hasattr(self.memory_service, 'close'):
            self.memory_service.close()
        # Verify mock close flags if needed for specific tests
        # self.assertTrue(self.session_service_driver_closed)
        # self.assertTrue(self.memory_service_driver_closed)


    def test_create_and_get_session(self):
        # Create a session
        sess = self.session_service.create_session(app_name="test_app", user_id="user123", state={"foo": "bar"})
        self.assertIsInstance(sess, Session)
        self.assertEqual(sess.app_name, "test_app")
        self.assertEqual(sess.user_id, "user123")
        self.assertEqual(sess.state.get("foo"), "bar")
        self.assertIsInstance(sess.last_update_time, float)

        # Retrieving the session should return the same data
        fetched = self.session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
        
        # --- Assertion that failed ---
        self.assertIsNotNone(fetched, "get_session should retrieve the created session") 
        # --- End Assertion ---
        
        if fetched: # Proceed only if fetched is not None
            self.assertEqual(fetched.id, sess.id)
            self.assertEqual(fetched.state, {"foo": "bar"}) 
            self.assertEqual(len(fetched.events), 0)
            self.assertIsInstance(fetched.last_update_time, float)
            self.assertAlmostEqual(fetched.last_update_time, sess.last_update_time, places=2)


    def test_append_event_state_update(self):
        sess = self.session_service.create_session(app_name="test_app", user_id="user123", state={"count": 1})
        original_last_update_time = sess.last_update_time

        evt_actions = EventActions(state_delta={"count": 2, "temp:note": "temp", "new_key": "value", "remove_this": None})
        evt_content = types.Content(parts=[types.Part(text="Hello world")])
        evt = Event(author="user", content=evt_content, actions=evt_actions)
        
        returned_event = self.session_service.append_event(sess, evt)
        self.assertEqual(returned_event, evt) 

        self.assertEqual(sess.state.get("count"), 2)
        self.assertNotIn("temp:note", sess.state)
        self.assertEqual(sess.state.get("new_key"), "value")
        self.assertNotIn("remove_this", sess.state)
        self.assertEqual(len(sess.events), 1)
        self.assertEqual(sess.events[0], evt)

        self.assertIsNotNone(evt.id) 
        evt_stored_in_mock = self._db["events"].get(evt.id)
        self.assertIsNotNone(evt_stored_in_mock)
        self.assertEqual(evt_stored_in_mock["text"], "Hello world") 

        self.assertNotAlmostEqual(sess.last_update_time, original_last_update_time, places=5)
        self.assertTrue(sess.last_update_time > original_last_update_time)
        
        fetched = self.session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
        
        # --- Assertion that failed ---
        self.assertIsNotNone(fetched, "get_session should retrieve the session after appending an event")
        # --- End Assertion ---

        if fetched: # Proceed only if fetched is not None
            self.assertEqual(fetched.state.get("count"), 2)
            self.assertIn("new_key", fetched.state)
            self.assertNotIn("remove_this", fetched.state)
            self.assertNotIn("temp:note", fetched.state) 

            self.assertEqual(len(fetched.events), 1)
            fetched_event = fetched.events[0]
            self.assertEqual(fetched_event.id, evt.id)
            self.assertEqual(fetched_event.author, evt.author)
            
            # Verify content_json hydration (P6)
            self.assertIsNotNone(fetched_event.content, "Fetched event content should not be None")
            self.assertIsInstance(fetched_event.content, types.Content, "Fetched event content should be of type types.Content")
            self.assertEqual(len(fetched_event.content.parts), 1, "Fetched event content should have one part")
            self.assertIsInstance(fetched_event.content.parts[0], types.Part, "Fetched event content part should be of type types.Part")
            self.assertEqual(fetched_event.content.parts[0].text, "Hello world", "Fetched event content text does not match")
            # Ensure full content object matches (if Content is a Pydantic model, equality should work)
            self.assertEqual(fetched_event.content, evt_content, "Full fetched event content should match original")

            # Check actions reconstruction
            # Note: Dummy EventActions might not have model_dump, adjust if using real ADK
            if fetched_event.actions and hasattr(evt_actions, 'model_dump'):
                 # Compare dicts if model_dump is available
                 self.assertEqual(fetched_event.actions.model_dump(), evt_actions.model_dump())
            elif fetched_event.actions:
                 # Fallback: Compare relevant attributes if model_dump is not available
                 self.assertEqual(fetched_event.actions.state_delta, evt_actions.state_delta)
                 self.assertEqual(fetched_event.actions.tool_calls, evt_actions.tool_calls)


    def test_search_memory_fulltext(self):
        sess = self.session_service.create_session(app_name="test_app", user_id="user123")
        evt1_content = types.Content(parts=[types.Part(text="Paris is the capital of France.")])
        evt1 = Event(author="user", content=evt1_content, actions=EventActions())
        self.session_service.append_event(sess, evt1)
        
        evt2_content = types.Content(parts=[types.Part(text="Sure, noted.")])
        evt2 = Event(author="agent", content=evt2_content, actions=EventActions())
        self.session_service.append_event(sess, evt2)
        
        self.memory_service.add_session_to_memory(sess)

        self.assertIn(evt1.id, self._db.get("memory_chunks", {}))
        self.assertEqual(self._db["memory_chunks"][evt1.id].get("text"), "Paris is the capital of France.")
        self.assertIn(evt2.id, self._db.get("memory_chunks", {}))
        
        response = self.memory_service.search_memory(app_name="test_app", user_id="user123", query="capital of France")
        
        self.assertIsInstance(response, SearchMemoryResponse)
        memories_list = response.memories
        self.assertIsNotNone(memories_list)
        self.assertGreaterEqual(len(memories_list), 1)
        
        found_event_text = False
        for mem_item in memories_list: 
            self.assertIsInstance(mem_item, MemoryResult) 
            if mem_item.session_id == sess.id:
                self.assertTrue(hasattr(mem_item, 'events'))
                for event_in_memory in mem_item.events:
                    self.assertIsInstance(event_in_memory, Event)
                    if event_in_memory.content and hasattr(event_in_memory.content, 'parts'):
                        for part in event_in_memory.content.parts:
                            if hasattr(part, 'text') and "Paris is the capital of France." in part.text:
                                found_event_text = True
                                break
                    if found_event_text: break
            if found_event_text: break
        
        self.assertTrue(found_event_text, "Memory search should retrieve the event with 'Paris is the capital of France.'")

    def test_list_sessions(self):
        self.session_service.create_session(app_name="app1", user_id="user1", state={"data": "session1"})
        self.session_service.create_session(app_name="app1", user_id="user1", state={"data": "session2"})
        self.session_service.create_session(app_name="app1", user_id="user2", state={"data": "session3"}) 

        listed_sessions_response = self.session_service.list_sessions(app_name="app1", user_id="user1")
        self.assertIsInstance(listed_sessions_response, ListSessionsResponse)
        self.assertEqual(len(listed_sessions_response.sessions), 2)
        session_ids = {s.id for s in listed_sessions_response.sessions}
        # Check if the sessions created for user1 are listed (IDs are random)
        user1_sessions_in_db = {sid for (a, u, sid), data in self._db["sessions"].items() if a == "app1" and u == "user1"}
        self.assertEqual(session_ids, user1_sessions_in_db)
        for s in listed_sessions_response.sessions:
            self.assertEqual(s.app_name, "app1")
            self.assertEqual(s.user_id, "user1")

    def test_list_events(self): # New test method
        sess = self.session_service.create_session(app_name="app_event_list", user_id="user_event_list")
        evt1_content = types.Content(parts=[types.Part(text="Event 1 for list")])
        evt1 = Event(author="user", content=evt1_content)
        self.session_service.append_event(sess, evt1)
        time.sleep(0.01) # Ensure timestamp difference

        evt2_content = types.Content(parts=[types.Part(text="Event 2 for list")])
        evt2 = Event(author="agent", content=evt2_content)
        self.session_service.append_event(sess, evt2)

        listed_events_response = self.session_service.list_events(
            app_name="app_event_list", user_id="user_event_list", session_id=sess.id
        )
        self.assertIsInstance(listed_events_response, ListEventsResponse)
        self.assertEqual(len(listed_events_response.events), 2)
        # Events should be sorted by timestamp
        self.assertEqual(listed_events_response.events[0].id, evt1.id)
        self.assertEqual(listed_events_response.events[1].id, evt2.id)

    def test_delete_session(self):
        sess_to_delete = self.session_service.create_session(app_name="app_del", user_id="user_del")
        evt_to_delete = Event(author="test", content=types.Content(parts=[types.Part(text="event to delete")]))
        self.session_service.append_event(sess_to_delete, evt_to_delete)
        
        # Check it exists before delete
        fetched_before = self.session_service.get_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)
        self.assertIsNotNone(fetched_before, "Session should exist before deletion")
        self.assertIn(evt_to_delete.id, self._db["events"], "Event should exist in mock DB before deletion") 

        # Delete
        self.session_service.delete_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)

        # Verify it's gone
        fetched_after = self.session_service.get_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)
        self.assertIsNone(fetched_after, "Session should be None after deletion")
        self.assertNotIn(evt_to_delete.id, self._db["events"], "Event should be removed from mock DB after deletion") 
        self.assertNotIn(("app_del", "user_del", sess_to_delete.id), self._db["sessions"], "Session key should be removed from mock DB")

    def test_append_event_stale_session_error(self):
        sess = self.session_service.create_session(app_name="test_stale", user_id="user_stale")
        
        mock_sess_key = (sess.app_name, sess.user_id, sess.id)
        # Ensure session exists in mock DB before modifying timestamp
        self.assertIn(mock_sess_key, self._db["sessions"])
        # Simulate external update by incrementing timestamp in mock DB
        self._db["sessions"][mock_sess_key]["last_update_time"] += 5000 

        evt_content = types.Content(parts=[types.Part(text="Attempt to append to stale session")])
        evt = Event(author="user", content=evt_content)
        
        # Attempt append with the original session object (stale timestamp)
        with self.assertRaises(StaleSessionError):
            self.session_service.append_event(sess, evt)

    def test_append_event_next_relationship(self):
        sess = self.session_service.create_session(app_name="test_next", user_id="user_next")
        
        evt1_content = types.Content(parts=[types.Part(text="Event 1")])
        evt1 = Event(author="user", content=evt1_content, timestamp=time.time()) 
        self.session_service.append_event(sess, evt1)
        time.sleep(0.01) 

        evt2_content = types.Content(parts=[types.Part(text="Event 2")])
        evt2 = Event(author="agent", content=evt2_content, timestamp=time.time())
        self.session_service.append_event(sess, evt2)
        time.sleep(0.01)

        evt3_content = types.Content(parts=[types.Part(text="Event 3")])
        evt3 = Event(author="user", content=evt3_content, timestamp=time.time())
        self.session_service.append_event(sess, evt3)

        # Verify NEXT relationships using the mock's internal state
        self.assertTrue(self._db["next_rels"].get((evt1.id, evt2.id)), "NEXT relationship from evt1 to evt2 should exist.")
        self.assertTrue(self._db["next_rels"].get((evt2.id, evt3.id)), "NEXT relationship from evt2 to evt3 should exist.")
        self.assertIsNone(self._db["next_rels"].get((evt1.id, evt3.id)), "Direct NEXT relationship from evt1 to evt3 should NOT exist.")

        # Verify using the specific Cypher query mentioned in the issue
        path_result = self.session_service._execute_read(
            "MATCH (a:Event {id:$e1})-[:NEXT]->()-[:NEXT]->(c:Event {id:$e3}) RETURN c",
            {"e1": evt1.id, "e3": evt3.id}
        )
        self.assertIsNotNone(path_result, "Query for e1->...->e3 should return a result.")
        self.assertGreaterEqual(len(path_result), 1, "Query for e1->...->e3 should find a path.")
        if path_result: 
             self.assertEqual(path_result[0]["c"]["id"], evt3.id, "The end of the e1->...->e3 path should be e3.")
 
    def test_append_event_with_tool_calls(self):
        """Verify ToolCall nodes and INVOKED_TOOL relationships are created."""
        sess = self.session_service.create_session(app_name="test_tool", user_id="user_tool")
        
        # Create FunctionCall instances (using the ToolCall alias) with 'args' field
        tool_call_1 = ToolCall(name="search_web", args={"query": "Neo4j graph database"}) # Use args=
        tool_call_2 = ToolCall(name="calculate", args={"expression": "2+2"}) # Use args=
        
        # Place FunctionCalls in event.content.parts as per current ADK standard
        evt_content = types.Content(parts=[
            types.Part(function_call=tool_call_1),
            types.Part(function_call=tool_call_2)
        ])
        # EventActions might still exist for state_delta, etc., but not for tool_calls
        evt_actions = EventActions()
        evt = Event(author="agent", content=evt_content, actions=evt_actions)

        # Append the event
        self.session_service.append_event(sess, evt)

        # Verify properties of stored ToolCall data in the mock DB
        # Since FunctionCall doesn't have a stable input ID, we find the stored data by name.
        # The _extract_tool_calls method generates a UUID for the Neo4j node ID.
        tc1_data = None
        tc2_data = None
        tc1_id = None
        tc2_id = None
        for tc_id, data in self._db["tool_calls"].items():
            if data["name"] == "search_web":
                tc1_data = data
                tc1_id = tc_id # Capture the generated ID
            elif data["name"] == "calculate":
                tc2_data = data
                tc2_id = tc_id # Capture the generated ID
        
        self.assertIsNotNone(tc1_data, "ToolCall data for 'search_web' should be found in mock DB")
        self.assertIsNotNone(tc1_id, "Generated ID for ToolCall 'search_web' should exist")
        if tc1_data:
            self.assertEqual(tc1_data["name"], "search_web")
            # Check parameters_json which should contain the 'args' dict dumped as JSON
            self.assertEqual(tc1_data["parameters_json"], json.dumps({"query": "Neo4j graph database"}))

        self.assertIsNotNone(tc2_data, "ToolCall data for 'calculate' should be found in mock DB")
        self.assertIsNotNone(tc2_id, "Generated ID for ToolCall 'calculate' should exist")
        if tc2_data:
            self.assertEqual(tc2_data["name"], "calculate")
            self.assertEqual(tc2_data["parameters_json"], json.dumps({"expression": "2+2"}))

        # Note: The mock doesn't explicitly track the INVOKED_TOOL relationship creation,
        # but we can infer it happened because the ToolCall nodes were added based on the
        # 'tool_calls_data' parameter passed to the mock write function.
        # A more detailed mock could track relationship creation if needed.

# Commenting out incomplete test from previous attempts
# def test_wrote_state_edge_creation(self):
#     """Test that WROTE_STATE edge is part of the Cypher query when state changes."""
#     pass

    def test_p7_shadow_app_state_creation_and_update(self):
        """Test P7: AppState node creation and update via append_event."""
        app_name = "p7_app_test"
        user_id = "p7_user"
        
        # Initial session, no app-specific state yet
        sess = self.session_service.create_session(app_name=app_name, user_id=user_id)
        self.assertNotIn(app_name, self._db["app_states"], "AppState should not exist yet.")

        # Event with app-specific delta
        evt_actions1 = EventActions(state_delta={"app:theme": "dark", "app:feature_flag_x": True, "session_key": "val1"})
        evt1 = Event(author="user", content=types.Content(parts=[types.Part(text="Set app theme")]), actions=evt_actions1)
        self.session_service.append_event(sess, evt1)

        self.assertIn(app_name, self._db["app_states"], "AppState should be created.")
        app_state_node = self._db["app_states"][app_name]
        app_state_data = json.loads(app_state_node["state_json"])
        self.assertEqual(app_state_data.get("app:theme"), "dark")
        self.assertTrue(app_state_data.get("app:feature_flag_x"))
        self.assertNotIn("session_key", app_state_data, "Session specific key should not be in AppState")
        self.assertIsNotNone(app_state_node.get("version"))
        original_app_version = app_state_node.get("version")

        # Event with another app-specific delta (update and new key)
        evt_actions2 = EventActions(state_delta={"app:theme": "light", "app:new_setting": "enabled"})
        evt2 = Event(author="user", content=types.Content(parts=[types.Part(text="Update app theme")]), actions=evt_actions2)
        time.sleep(0.001) # ensure timestamp difference for version
        self.session_service.append_event(sess, evt2)
        
        app_state_node_updated = self._db["app_states"][app_name]
        app_state_data_updated = json.loads(app_state_node_updated["state_json"])
        self.assertEqual(app_state_data_updated.get("app:theme"), "light")
        self.assertTrue(app_state_data_updated.get("app:feature_flag_x"), "Original app key should persist")
        self.assertEqual(app_state_data_updated.get("app:new_setting"), "enabled")
        self.assertTrue(app_state_node_updated.get("version") > original_app_version, "AppState version should be updated")

        # Event removing an app-specific key
        evt_actions3 = EventActions(state_delta={"app:feature_flag_x": None})
        evt3 = Event(author="user", content=types.Content(parts=[types.Part(text="Remove app feature flag")]), actions=evt_actions3)
        time.sleep(0.001)
        self.session_service.append_event(sess, evt3)

        app_state_node_final = self._db["app_states"][app_name]
        app_state_data_final = json.loads(app_state_node_final["state_json"])
        self.assertNotIn("app:feature_flag_x", app_state_data_final, "app:feature_flag_x should be removed from AppState")
        self.assertEqual(app_state_data_final.get("app:theme"), "light", "app:theme should still be light")


    def test_p7_shadow_user_state_creation_and_update(self):
        """Test P7: UserState node creation and update via append_event."""
        app_name = "p7_user_test_app"
        user_id = "p7_user1"
        user_key = (app_name, user_id)

        sess = self.session_service.create_session(app_name=app_name, user_id=user_id)
        self.assertNotIn(user_key, self._db["user_states"], "UserState should not exist yet.")

        evt_actions1 = EventActions(state_delta={"user:preference": "notifications_on", "user:language": "en", "session_detail": "abc"})
        evt1 = Event(author="user", content=types.Content(parts=[types.Part(text="Set user preference")]), actions=evt_actions1)
        self.session_service.append_event(sess, evt1)

        self.assertIn(user_key, self._db["user_states"], "UserState should be created.")
        user_state_node = self._db["user_states"][user_key]
        user_state_data = json.loads(user_state_node["state_json"])
        self.assertEqual(user_state_data.get("user:preference"), "notifications_on")
        self.assertEqual(user_state_data.get("user:language"), "en")
        self.assertNotIn("session_detail", user_state_data, "Session specific key should not be in UserState")
        original_user_version = user_state_node.get("version")

        evt_actions2 = EventActions(state_delta={"user:language": "fr", "user:last_login_ip": "1.2.3.4"})
        evt2 = Event(author="user", content=types.Content(parts=[types.Part(text="Update user language")]), actions=evt_actions2)
        time.sleep(0.001)
        self.session_service.append_event(sess, evt2)

        user_state_node_updated = self._db["user_states"][user_key]
        user_state_data_updated = json.loads(user_state_node_updated["state_json"])
        self.assertEqual(user_state_data_updated.get("user:preference"), "notifications_on")
        self.assertEqual(user_state_data_updated.get("user:language"), "fr")
        self.assertEqual(user_state_data_updated.get("user:last_login_ip"), "1.2.3.4")
        self.assertTrue(user_state_node_updated.get("version") > original_user_version)

    def test_p7_create_session_merges_shadow_states(self):
        """Test P7: create_session merges existing AppState and UserState."""
        app_name = "p7_merge_app"
        user_id1 = "p7_merge_user1"
        user_id2 = "p7_merge_user2" # Another user for distinct UserState

        # Pre-populate AppState
        self._db["app_states"][app_name] = {
            "state_json": json.dumps({"app:global_setting": "active", "app:common_pref": "default_app"}),
            "version": int(time.time() * 1000)
        }
        # Pre-populate UserState for user1
        user1_key = (app_name, user_id1)
        self._db["user_states"][user1_key] = {
            "state_json": json.dumps({"user:theme": "dark_user1", "user:custom_data": "user1_specific"}),
            "version": int(time.time() * 1000)
        }

        # Create session for user1, with some initial session-specific state
        # and an overlapping app-prefixed key to test merge behavior (shadow should win for its keys)
        sess1 = self.session_service.create_session(
            app_name=app_name,
            user_id=user_id1,
            state={"session_key": "val_s1", "app:common_pref": "session_override_app", "user:theme":"session_override_user"}
        )
        
        # Check merged state in the live Session object
        self.assertEqual(sess1.state.get("app:global_setting"), "active", "AppState global_setting missing")
        self.assertEqual(sess1.state.get("app:common_pref"), "default_app", "AppState common_pref should be merged from shadow")
        self.assertEqual(sess1.state.get("user:theme"), "dark_user1", "UserState theme should be merged from shadow")
        self.assertEqual(sess1.state.get("user:custom_data"), "user1_specific", "UserState custom_data missing")
        self.assertEqual(sess1.state.get("session_key"), "val_s1", "Session-specific key missing")
        
        # Verify the state_json stored for the session node in the mock DB also reflects this merge
        session_node_data = self._db["sessions"].get((app_name, user_id1, sess1.id))
        self.assertIsNotNone(session_node_data)
        stored_session_state = json.loads(session_node_data["state_json"])
        self.assertEqual(stored_session_state.get("app:global_setting"), "active")
        self.assertEqual(stored_session_state.get("app:common_pref"), "default_app")
        self.assertEqual(stored_session_state.get("user:theme"), "dark_user1")
        self.assertEqual(stored_session_state.get("user:custom_data"), "user1_specific")
        self.assertEqual(stored_session_state.get("session_key"), "val_s1")

        # Create session for user2 (should only get AppState, not UserState from user1)
        sess2 = self.session_service.create_session(app_name=app_name, user_id=user_id2, state={"session_key2": "val_s2"})
        self.assertEqual(sess2.state.get("app:global_setting"), "active")
        self.assertEqual(sess2.state.get("app:common_pref"), "default_app")
        self.assertIsNone(sess2.state.get("user:theme")) # Should not get user1's theme
        self.assertEqual(sess2.state.get("session_key2"), "val_s2")

    def test_p7_wrote_state_excludes_shadow_keys(self):
        """Test P7: WROTE_STATE relationships are not created for app: or user: prefixed keys."""
        app_name = "p7_wrote_state_app"
        user_id = "p7_wrote_state_user"
        sess = self.session_service.create_session(app_name=app_name, user_id=user_id)

        # Event with mixed state deltas
        evt_actions = EventActions(state_delta={
            "app:config": "new_app_config",
            "user:profile_status": "active_user",
            "session_data_point": "xyz123",
            "another_session_key": None # to test removal
        })
        evt = Event(author="user", content=types.Content(parts=[types.Part(text="Mixed state update")]), actions=evt_actions)
        
        # Clear last_write_query_string before append_event
        self.last_write_query_string = None
        self.session_service.append_event(sess, evt)
        
        # Inspect the Cypher query that was executed by the mock
        executed_cypher = self.last_write_query_string
        self.assertIsNotNone(executed_cypher)

        # Check that UNWIND for WROTE_STATE filters out app: and user: keys
        # This is a string check on the Cypher, which is a bit brittle but direct for this mock.
        # A more robust check would involve inspecting the parameters passed to the mock _execute_write,
        # specifically 'current_persisted_state_keys_values' after filtering.
        self.assertIn("UNWIND [k IN keys($current_persisted_state_keys_values) WHERE NOT (k STARTS WITH 'app:' OR k STARTS WITH 'user:')] AS k_session", executed_cypher)
        
        # Further check: The parameters passed to the mock for WROTE_STATE should reflect this.
        # The mock's fake_execute_write_session doesn't currently store the exact parameters for WROTE_STATE creation
        # in a way that's easily assertable here. The Cypher string check is the primary test for now.
        # If the mock was more detailed, we could check:
        # mock_params = self.session_service._last_params_for_append_event (if we stored it)
        # self.assertNotIn("app:config", mock_params["current_persisted_state_keys_values_for_wrote_state"])
        # self.assertNotIn("user:profile_status", mock_params["current_persisted_state_keys_values_for_wrote_state"])
        # self.assertIn("session_data_point", mock_params["current_persisted_state_keys_values_for_wrote_state"])
        
        # As a proxy, check that AppState and UserState were updated (implies keys were processed by shadow logic)
        self.assertIn(app_name, self._db["app_states"])
        self.assertEqual(json.loads(self._db["app_states"][app_name]["state_json"]).get("app:config"), "new_app_config")
        
        user_key = (app_name, user_id)
        self.assertIn(user_key, self._db["user_states"])
        self.assertEqual(json.loads(self._db["user_states"][user_key]["state_json"]).get("user:profile_status"), "active_user")

    def test_p7_shadow_app_state_creation_and_update(self):
        """Test P7: AppState node creation and update via append_event."""
        app_name = "p7_app_test"
        user_id = "p7_user"
        
        # Initial session, no app-specific state yet
        sess = self.session_service.create_session(app_name=app_name, user_id=user_id)
        self.assertNotIn(app_name, self._db["app_states"], "AppState should not exist yet.")

        # Event with app-specific delta
        evt_actions1 = EventActions(state_delta={"app:theme": "dark", "app:feature_flag_x": True, "session_key": "val1"})
        evt1 = Event(author="user", content=types.Content(parts=[types.Part(text="Set app theme")]), actions=evt_actions1)
        self.session_service.append_event(sess, evt1)

        self.assertIn(app_name, self._db["app_states"], "AppState should be created.")
        app_state_node = self._db["app_states"][app_name]
        app_state_data = json.loads(app_state_node["state_json"])
        self.assertEqual(app_state_data.get("app:theme"), "dark")
        self.assertTrue(app_state_data.get("app:feature_flag_x"))
        self.assertNotIn("session_key", app_state_data, "Session specific key should not be in AppState")
        self.assertIsNotNone(app_state_node.get("version"))
        original_app_version = app_state_node.get("version")

        # Event with another app-specific delta (update and new key)
        evt_actions2 = EventActions(state_delta={"app:theme": "light", "app:new_setting": "enabled"})
        evt2 = Event(author="user", content=types.Content(parts=[types.Part(text="Update app theme")]), actions=evt_actions2)
        time.sleep(0.001) # ensure timestamp difference for version
        self.session_service.append_event(sess, evt2)
        
        app_state_node_updated = self._db["app_states"][app_name]
        app_state_data_updated = json.loads(app_state_node_updated["state_json"])
        self.assertEqual(app_state_data_updated.get("app:theme"), "light")
        self.assertTrue(app_state_data_updated.get("app:feature_flag_x"), "Original app key should persist")
        self.assertEqual(app_state_data_updated.get("app:new_setting"), "enabled")
        self.assertTrue(app_state_node_updated.get("version") > original_app_version, "AppState version should be updated")

        # Event removing an app-specific key
        evt_actions3 = EventActions(state_delta={"app:feature_flag_x": None})
        evt3 = Event(author="user", content=types.Content(parts=[types.Part(text="Remove app feature flag")]), actions=evt_actions3)
        time.sleep(0.001)
        self.session_service.append_event(sess, evt3)

        app_state_node_final = self._db["app_states"][app_name]
        app_state_data_final = json.loads(app_state_node_final["state_json"])
        self.assertNotIn("app:feature_flag_x", app_state_data_final, "app:feature_flag_x should be removed from AppState")
        self.assertEqual(app_state_data_final.get("app:theme"), "light", "app:theme should still be light")


    def test_p7_shadow_user_state_creation_and_update(self):
        """Test P7: UserState node creation and update via append_event."""
        app_name = "p7_user_test_app"
        user_id = "p7_user1"
        user_key = (app_name, user_id)

        sess = self.session_service.create_session(app_name=app_name, user_id=user_id)
        self.assertNotIn(user_key, self._db["user_states"], "UserState should not exist yet.")

        evt_actions1 = EventActions(state_delta={"user:preference": "notifications_on", "user:language": "en", "session_detail": "abc"})
        evt1 = Event(author="user", content=types.Content(parts=[types.Part(text="Set user preference")]), actions=evt_actions1)
        self.session_service.append_event(sess, evt1)

        self.assertIn(user_key, self._db["user_states"], "UserState should be created.")
        user_state_node = self._db["user_states"][user_key]
        user_state_data = json.loads(user_state_node["state_json"])
        self.assertEqual(user_state_data.get("user:preference"), "notifications_on")
        self.assertEqual(user_state_data.get("user:language"), "en")
        self.assertNotIn("session_detail", user_state_data, "Session specific key should not be in UserState")
        original_user_version = user_state_node.get("version")

        evt_actions2 = EventActions(state_delta={"user:language": "fr", "user:last_login_ip": "1.2.3.4"})
        evt2 = Event(author="user", content=types.Content(parts=[types.Part(text="Update user language")]), actions=evt_actions2)
        time.sleep(0.001)
        self.session_service.append_event(sess, evt2)

        user_state_node_updated = self._db["user_states"][user_key]
        user_state_data_updated = json.loads(user_state_node_updated["state_json"])
        self.assertEqual(user_state_data_updated.get("user:preference"), "notifications_on")
        self.assertEqual(user_state_data_updated.get("user:language"), "fr")
        self.assertEqual(user_state_data_updated.get("user:last_login_ip"), "1.2.3.4")
        self.assertTrue(user_state_node_updated.get("version") > original_user_version)

    def test_p7_create_session_merges_shadow_states(self):
        """Test P7: create_session merges existing AppState and UserState."""
        app_name = "p7_merge_app"
        user_id1 = "p7_merge_user1"
        user_id2 = "p7_merge_user2" # Another user for distinct UserState

        # Pre-populate AppState
        self._db["app_states"][app_name] = {
            "state_json": json.dumps({"app:global_setting": "active", "app:common_pref": "default_app"}),
            "version": int(time.time() * 1000)
        }
        # Pre-populate UserState for user1
        user1_key = (app_name, user_id1)
        self._db["user_states"][user1_key] = {
            "state_json": json.dumps({"user:theme": "dark_user1", "user:custom_data": "user1_specific"}),
            "version": int(time.time() * 1000)
        }

        # Create session for user1, with some initial session-specific state
        # and an overlapping app-prefixed key to test merge behavior (shadow should win for its keys)
        sess1 = self.session_service.create_session(
            app_name=app_name,
            user_id=user_id1,
            state={"session_key": "val_s1", "app:common_pref": "session_override_app", "user:theme":"session_override_user"}
        )
        
        # Check merged state in the live Session object
        self.assertEqual(sess1.state.get("app:global_setting"), "active", "AppState global_setting missing")
        self.assertEqual(sess1.state.get("app:common_pref"), "default_app", "AppState common_pref should be merged from shadow")
        self.assertEqual(sess1.state.get("user:theme"), "dark_user1", "UserState theme should be merged from shadow")
        self.assertEqual(sess1.state.get("user:custom_data"), "user1_specific", "UserState custom_data missing")
        self.assertEqual(sess1.state.get("session_key"), "val_s1", "Session-specific key missing")
        
        # Verify the state_json stored for the session node in the mock DB also reflects this merge
        session_node_data = self._db["sessions"].get((app_name, user_id1, sess1.id))
        self.assertIsNotNone(session_node_data)
        stored_session_state = json.loads(session_node_data["state_json"])
        self.assertEqual(stored_session_state.get("app:global_setting"), "active")
        self.assertEqual(stored_session_state.get("app:common_pref"), "default_app")
        self.assertEqual(stored_session_state.get("user:theme"), "dark_user1")
        self.assertEqual(stored_session_state.get("user:custom_data"), "user1_specific")
        self.assertEqual(stored_session_state.get("session_key"), "val_s1")

        # Create session for user2 (should only get AppState, not UserState from user1)
        sess2 = self.session_service.create_session(app_name=app_name, user_id=user_id2, state={"session_key2": "val_s2"})
        self.assertEqual(sess2.state.get("app:global_setting"), "active")
        self.assertEqual(sess2.state.get("app:common_pref"), "default_app")
        self.assertIsNone(sess2.state.get("user:theme")) # Should not get user1's theme
        self.assertEqual(sess2.state.get("session_key2"), "val_s2")

    def test_p7_wrote_state_excludes_shadow_keys(self):
        """Test P7: WROTE_STATE relationships are not created for app: or user: prefixed keys."""
        app_name = "p7_wrote_state_app"
        user_id = "p7_wrote_state_user"
        sess = self.session_service.create_session(app_name=app_name, user_id=user_id)

        # Event with mixed state deltas
        evt_actions = EventActions(state_delta={
            "app:config": "new_app_config",
            "user:profile_status": "active_user",
            "session_data_point": "xyz123",
            "another_session_key": None # to test removal
        })
        evt = Event(author="user", content=types.Content(parts=[types.Part(text="Mixed state update")]), actions=evt_actions)
        
        # Clear last_write_query_string before append_event
        self.last_write_query_string = None
        self.session_service.append_event(sess, evt)
        
        # Inspect the Cypher query that was executed by the mock
        executed_cypher = self.last_write_query_string
        self.assertIsNotNone(executed_cypher)

        # Check that UNWIND for WROTE_STATE filters out app: and user: keys
        # This is a string check on the Cypher, which is a bit brittle but direct for this mock.
        # A more robust check would involve inspecting the parameters passed to the mock _execute_write,
        # specifically 'current_persisted_state_keys_values' after filtering.
        self.assertIn("UNWIND [k IN keys($current_persisted_state_keys_values) WHERE NOT (k STARTS WITH 'app:' OR k STARTS WITH 'user:')] AS k_session", executed_cypher)
        
        # Further check: The parameters passed to the mock for WROTE_STATE should reflect this.
        # The mock's fake_execute_write_session doesn't currently store the exact parameters for WROTE_STATE creation
        # in a way that's easily assertable here. The Cypher string check is the primary test for now.
        # If the mock was more detailed, we could check:
        # mock_params = self.session_service._last_params_for_append_event (if we stored it)
        # self.assertNotIn("app:config", mock_params["current_persisted_state_keys_values_for_wrote_state"])
        # self.assertNotIn("user:profile_status", mock_params["current_persisted_state_keys_values_for_wrote_state"])
        # self.assertIn("session_data_point", mock_params["current_persisted_state_keys_values_for_wrote_state"])
        
        # As a proxy, check that AppState and UserState were updated (implies keys were processed by shadow logic)
        self.assertIn(app_name, self._db["app_states"])
        self.assertEqual(json.loads(self._db["app_states"][app_name]["state_json"]).get("app:config"), "new_app_config")
        
        user_key = (app_name, user_id)
        self.assertIn(user_key, self._db["user_states"])
        self.assertEqual(json.loads(self._db["user_states"][user_key]["state_json"]).get("user:profile_status"), "active_user")
