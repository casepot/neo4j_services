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
from neo4j_adk_services.src.neo4j_session_service import Neo4jSessionService
from neo4j_adk_services.src.neo4j_memory_service import Neo4jMemoryService
# Assuming SearchMemoryResponse would be part of google.adk.memory
try:
    from google.adk.memory import SearchMemoryResponse
except ImportError:
    class SearchMemoryResponse: # Dummy for tests if not found
        def __init__(self, results=None):
            self.results = results or []


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
        self._db = {"sessions": {}, "events": {}} # Stores session and event data for mock
        
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
            if "CREATE CONSTRAINT IF NOT EXISTS FOR (t:ToolCall) REQUIRE t.id IS UNIQUE" in query:
                return []
            if "CREATE CONSTRAINT IF NOT EXISTS ON (s:Session) ASSERT (s.app_name, s.user_id, s.id) IS UNIQUE" in query:
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
                    "last_update_time": params["timestamp"]
                    # 'events' list will be implicitly associated by session_key in _db["events"]
                }
                # Return what the actual query returns
                return [{
                    "app_name": app_name, "user_id": user_id, "id": sid,
                    "state_json": params["state_json"], "last_update_time": params["timestamp"]
                }]

            # Mock for append_event
            # CREATE (e:Event)-[:OF_SESSION]->(s) SET e = $event_props
            # SET s.state_json = $new_session_state_json, s.last_update_time = $event_timestamp
            elif "CREATE (e:Event)-[:OF_SESSION]->(s)" in query and "SET e = $event_props" in query:
                event_props = params["event_props"]
                eid = event_props["id"]
                sid = params["session_id"]
                app_name = params["app_name"]
                user_id = params["user_id"]
                
                # Store event
                self._db["events"][eid] = {
                    "session_key": (app_name, user_id, sid),
                    **event_props # Store all properties from event_props
                }
                
                # Update session
                sess_key = (app_name, user_id, sid)
                if sess_key in self._db["sessions"]:
                    self._db["sessions"][sess_key]["state_json"] = params["new_session_state_json"]
                    self._db["sessions"][sess_key]["last_update_time"] = params["event_timestamp"]
                
                # Simulate WROTE_STATE, NEXT, INVOKED_TOOL if needed for more detailed tests later
                # For now, just ensuring event and session update is primary.
                return [{"event_id": eid}] # Actual query returns e.id

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
                
                # Sort events by timestamp as the actual query does
                collected_events_data.sort(key=lambda e: e.get("timestamp", 0))
                
                # The actual query returns `s` (the session node) and `collect(e) AS events`
                # `s_node` in get_session becomes `record['s']`
                # `events_list` becomes `record['events']`
                return [{"s": session_node_data, "events": collected_events_data}]

            # For list_sessions: MATCH (s:Session {app_name: $app_name, user_id: $user_id}) RETURN s.id AS session_id, s.last_update_time AS last_update
            elif "RETURN s.id AS session_id, s.last_update_time AS last_update" in query:
                # list_sessions query
                app, user = params["app_name"], params["user_id"]
                result = []
                for (a, u, sid), sess_data in self._db["sessions"].items():
                    if a == app and u == user:
                        result.append({"session_id": sid, "last_update": sess_data["last_update_time"]})
                return result
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
                        # The test expects session_id. We need to retrieve this.
                        # This requires linking MemoryChunk back to a Session or storing session_id on MemoryChunk.
                        # The current MemoryChunk mock stores eid. We need to find the session via eid.
                        original_session_id = None
                        for _eid_event, event_detail in self._db.get("events", {}).items():
                            if _eid_event == chunk_data.get("eid"):
                                original_session_id = event_detail.get("session_key")[2] # (app, user, sid)
                                break
                        
                        mock_results.append({
                            # "session_id": chunk_data.get("session_id"), # If MemoryChunk stored session_id
                            "session_id": original_session_id, # Derived for the test
                            "event_id": chunk_data.get("eid"), # Keep event_id as per query
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
        # Retrieving the session should return the same data
        fetched = self.session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, sess.id)
        self.assertEqual(fetched.state, {"foo": "bar"}) # State is JSON dumped and loaded, should be equal
        self.assertEqual(len(fetched.events), 0)  # no events yet

    def test_append_event_state_update(self):
        sess = self.session_service.create_session(app_name="test_app", user_id="user123", state={"count": 1})
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

        # Session last_update_time should match event timestamp
        self.assertAlmostEqual(sess.last_update_time, evt.timestamp, places=5)
        
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
        
        # Verify the memory search response contains the expected snippet
        # Verify the memory search response contains the expected snippet
        # SearchMemoryResponse uses 'memories' attribute
        if isinstance(response, SearchMemoryResponse):
            self.assertTrue(hasattr(response, 'memories'), "Response object should have a 'memories' attribute")
            memories_list = response.memories
        elif isinstance(response, dict): # Fallback case in the service
            self.assertIn('memories', response, "Response dict should have a 'memories' key")
            memories_list = response['memories']
        else:
            self.fail(f"Unexpected response type: {type(response)}")

        self.assertGreaterEqual(len(memories_list), 1, "Should find at least one matching session")
        
        found_snippet = False
        for mem_item_data in memories_list: # mem_item_data is expected to be a dict like {"session_id": ..., "snippets": ...}
            # The MemoryResult objects themselves might be dicts or objects depending on ADK version/Pydantic
            # For this test, the service mock returns dicts for snippets.
            current_session_id = None
            current_snippets = []
            if isinstance(mem_item_data, dict):
                current_session_id = mem_item_data.get("session_id")
                current_snippets = mem_item_data.get("snippets", [])
            # If MemoryResult is an object, adjust access accordingly, e.g. mem_item_data.session_id
            # else:
            #     current_session_id = getattr(mem_item_data, "session_id", None)
            #     current_snippets = getattr(mem_item_data, "snippets", [])


            if current_session_id == sess.id:
                for snippet in current_snippets:
                    if "Paris is the capital of France." in snippet:
                        found_snippet = True
                        break
            if found_snippet:
                break
        
        self.assertTrue(found_snippet, "Memory search should retrieve the expected snippet from session memory")

    def test_list_sessions(self):
        self.session_service.create_session(app_name="app1", user_id="user1", state={"data": "session1"})
        self.session_service.create_session(app_name="app1", user_id="user1", state={"data": "session2"})
        self.session_service.create_session(app_name="app1", user_id="user2", state={"data": "session3"}) # Different user

        listed_sessions = self.session_service.list_sessions(app_name="app1", user_id="user1")
        self.assertEqual(len(listed_sessions), 2)
        for s in listed_sessions:
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

if __name__ == '__main__':
    unittest.main()