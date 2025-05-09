import pytest # For using pytest fixtures and async capabilities
import pytest_asyncio # For async fixtures
import time
import uuid
import json
import asyncio # For async operations in tests

from neo4j import AsyncGraphDatabase, GraphDatabase # For direct DB interaction in tests if needed
# from neo4j.exceptions import StaleDataError as Neo4jStaleDataError # Neo4j's own exception - Removed as it causes ImportError

from google.adk.sessions import Session
from google.adk.sessions.base_session_service import ListSessionsResponse, ListEventsResponse
from google.adk.events import Event, EventActions
from google.genai import types
from google.genai.types import FunctionCall as ToolCall
from google.adk.memory.base_memory_service import SearchMemoryResponse, MemoryResult

from neo4j_adk_services.src.neo4j_session_service import Neo4jSessionService, StaleSessionError
from neo4j_adk_services.src.neo4j_memory_service import Neo4jMemoryService

# Default embedding dimension for tests, ensure this matches what MemoryService might expect
TEST_VECTOR_DIMENSION = 128

# Dummy embedding function for tests
def get_test_embedding(text: str) -> list[float]:
    # Simple hash-based embedding for testing purposes
    # Ensure the length matches TEST_VECTOR_DIMENSION
    embedding = [0.0] * TEST_VECTOR_DIMENSION
    for i, char_code in enumerate(map(ord, text)):
        if i < TEST_VECTOR_DIMENSION:
            embedding[i] = (char_code % 256) / 255.0
    return embedding

@pytest.fixture(scope="function")
def clear_db(neo4j_uri, neo4j_auth): # Changed to synchronous fixture
    """Clears all data from the Neo4j database before each test function."""
    driver = None
    try:
        # Expecting plain bolt:// URI as TLS will be disabled in the container
        driver = GraphDatabase.driver(
            neo4j_uri, # Use the provided URI directly (expected to be bolt://)
            auth=neo4j_auth
            # No explicit encryption flags, relying on plain bolt connection
        )
        with driver.session(database="neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Optionally, remove indexes and constraints if they interfere,
            # but services should handle "IF NOT EXISTS"
            # Example: session.run("CALL apoc.schema.assert({}, {})") to clear all
        print("Database cleared for test function.")
    except Exception as e:
        print(f"Error clearing database: {e}")
        # Depending on strictness, you might want to fail the test here
        # pytest.fail(f"Failed to clear database: {e}")
    finally:
        if driver:
            driver.close()
    # Yield control to the test
    yield
    # No post-yield cleanup needed here as we clear before each test


@pytest.fixture(scope="function")
def session_service(neo4j_uri, neo4j_auth, clear_db): # Depends on clear_db to run first
    """Provides a Neo4jSessionService instance connected to the testcontainer DB."""
    service = Neo4jSessionService(uri=neo4j_uri, user=neo4j_auth[0], password=neo4j_auth[1], database="neo4j")
    yield service
    service.close()

@pytest_asyncio.fixture(scope="function")
async def memory_service(neo4j_uri, neo4j_async_uri, neo4j_auth, clear_db): # Depends on clear_db
    """Provides a Neo4jMemoryService instance connected to the testcontainer DB."""
    # Prefer the synchronous value; fall back to the coroutine
    uri_to_use = neo4j_uri or neo4j_async_uri
    if asyncio.iscoroutine(uri_to_use):
        uri_to_use = await uri_to_use            # resolve coroutine → str
    if not uri_to_use:
        raise RuntimeError("Unable to determine Neo4j bolt URI for memory_service")

    service = Neo4jMemoryService(
        uri=uri_to_use,
        user=neo4j_auth[0],
        password=neo4j_auth[1],
        database="neo4j",
        embedding_function=get_test_embedding,
        vector_dimension=TEST_VECTOR_DIMENSION
    )
    # Ensure indexes are created by calling an async method that triggers it
    try:
        await service._ensure_indexes() # Call directly if accessible and safe for tests
    except Exception as e:
        print(f"Error ensuring indexes for memory_service fixture: {e}")
        # pytest.fail(f"Failed to ensure indexes for memory_service: {e}")

    yield service
    await service.close()


# --- Test Functions (pytest style) ---

def test_create_and_get_session(session_service: Neo4jSessionService):
    """Tests basic session creation and retrieval."""
    sess = session_service.create_session(app_name="test_app", user_id="user123", state={"foo": "bar"})
    assert isinstance(sess, Session)
    assert sess.app_name == "test_app"
    assert sess.user_id == "user123"
    assert sess.state.get("foo") == "bar"
    assert isinstance(sess.last_update_time, float)

    fetched = session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
    assert fetched is not None, "get_session should retrieve the created session"
    if fetched:
        assert fetched.id == sess.id
        assert fetched.state == {"foo": "bar"}
        assert len(fetched.events) == 0
        assert isinstance(fetched.last_update_time, float)
        assert fetched.last_update_time == pytest.approx(sess.last_update_time, rel=1e-2)


def test_append_event_state_update(session_service: Neo4jSessionService):
    """Tests appending an event and verifying state updates."""
    sess = session_service.create_session(app_name="test_app", user_id="user123", state={"count": 1})
    original_last_update_time = sess.last_update_time

    evt_actions = EventActions(state_delta={"count": 2, "temp:note": "temp", "new_key": "value", "remove_this": None})
    evt_content = types.Content(parts=[types.Part(text="Hello world")])
    evt = Event(author="user", content=evt_content, actions=evt_actions)

    returned_event = session_service.append_event(sess, evt)
    assert returned_event.id == evt.id # Compare by ID

    assert sess.state.get("count") == 2
    assert "temp:note" not in sess.state
    assert sess.state.get("new_key") == "value"
    assert "remove_this" not in sess.state
    assert len(sess.events) == 1
    assert sess.events[0].id == evt.id # Compare by ID

    assert evt.id is not None
    assert sess.last_update_time != pytest.approx(original_last_update_time, abs=1e-3)
    assert sess.last_update_time > original_last_update_time

    fetched = session_service.get_session(app_name="test_app", user_id="user123", session_id=sess.id)
    assert fetched is not None, "get_session should retrieve the session after appending an event"
    if fetched:
        assert fetched.state.get("count") == 2
        assert "new_key" in fetched.state
        assert "remove_this" not in fetched.state
        assert "temp:note" not in fetched.state

        assert len(fetched.events) == 1
        fetched_event = fetched.events[0]
        assert fetched_event.id == evt.id
        assert fetched_event.author == evt.author

        assert fetched_event.content is not None
        assert isinstance(fetched_event.content, types.Content)
        assert len(fetched_event.content.parts) == 1
        assert isinstance(fetched_event.content.parts[0], types.Part)
        assert fetched_event.content.parts[0].text == "Hello world"
        assert fetched_event.content == evt_content

        if fetched_event.actions and hasattr(evt_actions, 'model_dump'):
             assert fetched_event.actions.model_dump() == evt_actions.model_dump()
        elif fetched_event.actions:
             assert fetched_event.actions.state_delta == evt_actions.state_delta


@pytest.mark.asyncio
async def test_search_memory_fulltext_and_vector_roundtrip(session_service: Neo4jSessionService, memory_service: Neo4jMemoryService):
    """Tests adding a session to memory and searching via full-text and vector."""
    app_name_mem = "test_mem_app"
    user_id_mem = "user_mem123"
    sess = session_service.create_session(app_name=app_name_mem, user_id=user_id_mem)

    text1 = "Paris is the beautiful capital of France."
    text2 = "Neo4j is a graph database."

    evt1_content = types.Content(parts=[types.Part(text=text1)])
    evt1 = Event(author="user", content=evt1_content, actions=EventActions(), timestamp=time.time())
    session_service.append_event(sess, evt1)
    await asyncio.sleep(0.01) # Ensure distinct timestamps

    evt2_content = types.Content(parts=[types.Part(text=text2)])
    evt2 = Event(author="agent", content=evt2_content, actions=EventActions(), timestamp=time.time())
    session_service.append_event(sess, evt2)

    refreshed_session = session_service.get_session(app_name=app_name_mem, user_id=user_id_mem, session_id=sess.id)
    assert refreshed_session is not None
    assert len(refreshed_session.events) == 2

    await memory_service.add_session_to_memory(refreshed_session)

    # Test full-text search
    response_ft = await memory_service.search_memory(app_name=app_name_mem, user_id=user_id_mem, query="capital of France")
    assert isinstance(response_ft, SearchMemoryResponse)
    assert len(response_ft.memories) >= 1, "Full-text search should find at least one memory item."

    found_text1_ft = any(
        text1 in part.text
        for mem_item in response_ft.memories if mem_item.session_id == sess.id
        for event_in_memory in mem_item.events
        for part in (event_in_memory.content.parts if event_in_memory.content and hasattr(event_in_memory.content, 'parts') else [])
    )
    assert found_text1_ft, "Full-text search should retrieve the event with 'Paris is the beautiful capital of France.'"

    # Test vector search
    query_sem = "Paris"
    response_vec = await memory_service.search_memory(app_name=app_name_mem, user_id=user_id_mem, query=query_sem)
    assert isinstance(response_vec, SearchMemoryResponse)
    assert len(response_vec.memories) >= 1, "Vector search should find at least one memory item for 'Paris'."

    found_text1_vec = any(
        text1 in part.text
        for mem_item in response_vec.memories if mem_item.session_id == sess.id
        for event_in_memory in mem_item.events
        for part in (event_in_memory.content.parts if event_in_memory.content and hasattr(event_in_memory.content, 'parts') else [])
    )
    assert found_text1_vec, "Vector search should retrieve the event containing 'Paris' based on semantic query."

    assert memory_service._mem_idx_dim_cache is not None, "Vector index dimension should be cached after search."
    assert memory_service._mem_idx_dim_cache == TEST_VECTOR_DIMENSION


def test_list_sessions(session_service: Neo4jSessionService):
    """Tests listing sessions for a user."""
    session_service.create_session(app_name="app1", user_id="user1", state={"data": "session1"})
    session_service.create_session(app_name="app1", user_id="user1", state={"data": "session2"})
    session_service.create_session(app_name="app1", user_id="user2", state={"data": "session3"}) # Different user

    listed_sessions_response = session_service.list_sessions(app_name="app1", user_id="user1")
    assert isinstance(listed_sessions_response, ListSessionsResponse)
    assert len(listed_sessions_response.sessions) == 2
    # Check user and app names
    for sess_info in listed_sessions_response.sessions:
        assert sess_info.app_name == "app1"
        assert sess_info.user_id == "user1"


@pytest.mark.asyncio
async def test_list_events_with_paging(session_service: Neo4jSessionService):
    """Tests listing events with limit and skip (simulated via 'after')."""
    sess = session_service.create_session(app_name="app_event_list", user_id="user_event_list")
    all_event_ids = []
    for i in range(5):
        evt_content = types.Content(parts=[types.Part(text=f"Event {i} for list")])
        evt = Event(author="user", content=evt_content, timestamp=time.time())
        session_service.append_event(sess, evt)
        all_event_ids.append(evt.id)
        await asyncio.sleep(0.001) # Ensure unique timestamps

    # Test limit
    listed_events_limit_2 = session_service.list_events(
        app_name="app_event_list", user_id="user_event_list", session_id=sess.id, limit=2
    )
    assert isinstance(listed_events_limit_2, ListEventsResponse)
    assert len(listed_events_limit_2.events) == 2
    assert listed_events_limit_2.events[0].id == all_event_ids[0]
    assert listed_events_limit_2.events[1].id == all_event_ids[1]

    # Test limit and after (as offset)
    listed_events_after_2_limit_2 = session_service.list_events(
        app_name="app_event_list", user_id="user_event_list", session_id=sess.id, after="2", limit=2 # "2" means skip 2
    )
    assert isinstance(listed_events_after_2_limit_2, ListEventsResponse)
    assert len(listed_events_after_2_limit_2.events) == 2
    assert listed_events_after_2_limit_2.events[0].id == all_event_ids[2]
    assert listed_events_after_2_limit_2.events[1].id == all_event_ids[3]

    # Test getting all events
    listed_events_all = session_service.list_events(
        app_name="app_event_list", user_id="user_event_list", session_id=sess.id, limit=10
    )
    assert len(listed_events_all.events) == 5


def test_delete_session(session_service: Neo4jSessionService):
    """Tests deleting a session and its associated data."""
    sess_to_delete = session_service.create_session(app_name="app_del", user_id="user_del")
    evt_to_delete_content = types.Content(parts=[types.Part(text="event to delete")])
    evt_to_delete = Event(author="test", content=evt_to_delete_content)
    session_service.append_event(sess_to_delete, evt_to_delete)

    fetched_before = session_service.get_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)
    assert fetched_before is not None, "Session should exist before deletion"

    session_service.delete_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)

    fetched_after = session_service.get_session(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)
    assert fetched_after is None, "Session should be None after deletion"

    events_after_delete = session_service.list_events(app_name="app_del", user_id="user_del", session_id=sess_to_delete.id)
    assert len(events_after_delete.events) == 0, "Events should be deleted with the session"


def test_append_event_stale_session_error(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests that StaleSessionError is raised on optimistic locking failure."""
    sess = session_service.create_session(app_name="test_stale", user_id="user_stale")

    # Simulate external update
    sync_driver_modifier = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_modifier.session(database="neo4j") as db_sess_modifier:
        db_sess_modifier.run(
            "MATCH (s:Session {id: $sid}) SET s.last_update_time = s.last_update_time + 5000",
            sid=sess.id
        )
    sync_driver_modifier.close()

    evt_content = types.Content(parts=[types.Part(text="Attempt to append to stale session")])
    evt = Event(author="user", content=evt_content)

    with pytest.raises(StaleSessionError):
        session_service.append_event(sess, evt)


@pytest.mark.asyncio
async def test_append_event_next_relationship(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests the creation of NEXT relationships between events."""
    sess = session_service.create_session(app_name="test_next", user_id="user_next")

    evt1_content = types.Content(parts=[types.Part(text="Event 1")])
    evt1 = Event(author="user", content=evt1_content, timestamp=time.time())
    session_service.append_event(sess, evt1)
    await asyncio.sleep(0.01)

    evt2_content = types.Content(parts=[types.Part(text="Event 2")])
    evt2 = Event(author="agent", content=evt2_content, timestamp=time.time())
    session_service.append_event(sess, evt2)
    await asyncio.sleep(0.01)

    evt3_content = types.Content(parts=[types.Part(text="Event 3")])
    evt3 = Event(author="user", content=evt3_content, timestamp=time.time())
    session_service.append_event(sess, evt3)

    # Verify NEXT relationships directly in the DB
    sync_driver_checker = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        res1_2 = db_sess_checker.run("MATCH (:Event {id:$e1id})-[:NEXT]->(:Event {id:$e2id}) RETURN count(*) AS c", e1id=evt1.id, e2id=evt2.id).single()
        assert res1_2["c"] == 1, "NEXT relationship from evt1 to evt2 should exist."

        res2_3 = db_sess_checker.run("MATCH (:Event {id:$e2id})-[:NEXT]->(:Event {id:$e3id}) RETURN count(*) AS c", e2id=evt2.id, e3id=evt3.id).single()
        assert res2_3["c"] == 1, "NEXT relationship from evt2 to evt3 should exist."

        res1_3 = db_sess_checker.run("MATCH (:Event {id:$e1id})-[:NEXT]->(:Event {id:$e3id}) RETURN count(*) AS c", e1id=evt1.id, e3id=evt3.id).single()
        assert res1_3["c"] == 0, "Direct NEXT relationship from evt1 to evt3 should NOT exist."

        path_res = db_sess_checker.run(
            "MATCH p=(e1:Event {id:$e1id})-[:NEXT*2]->(e3:Event {id:$e3id}) WHERE (e1)-[:NEXT]->()-[:NEXT]->(e3) RETURN count(p) AS c",
            e1id=evt1.id, e3id=evt3.id
        ).single()
        assert path_res["c"] == 1, "Path e1->e2->e3 should exist."
    sync_driver_checker.close()


def test_append_event_with_tool_calls(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests creation of ToolCall nodes and INVOKED_TOOL relationships."""
    sess = session_service.create_session(app_name="test_tool", user_id="user_tool")

    tool_call_1 = ToolCall(name="search_web", args={"query": "Neo4j graph database"})
    tool_call_2 = ToolCall(name="calculate", args={"expression": "2+2"})

    evt_content = types.Content(parts=[
        types.Part(function_call=tool_call_1),
        types.Part(function_call=tool_call_2)
    ])
    evt_actions = EventActions()
    evt = Event(author="agent", content=evt_content, actions=evt_actions)
    session_service.append_event(sess, evt)

    # Verify ToolCall nodes and relationships in the DB
    sync_driver_checker = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        tc1_node_data = db_sess_checker.run(
            "MATCH (tc:ToolCall {name: 'search_web'}) RETURN tc.id AS id, tc.parameters_json AS params"
        ).single()
        assert tc1_node_data is not None, "ToolCall 'search_web' not found in DB."
        if tc1_node_data:
            assert json.loads(tc1_node_data["params"]) == {"query": "Neo4j graph database"}
            rel1_check = db_sess_checker.run(
                "MATCH (:Event {id:$eid})-[:INVOKED_TOOL]->(:ToolCall {id:$tcid}) RETURN count(*) AS c",
                eid=evt.id, tcid=tc1_node_data["id"]
            ).single()
            assert rel1_check["c"] == 1, "INVOKED_TOOL relationship missing for search_web."

        tc2_node_data = db_sess_checker.run(
            "MATCH (tc:ToolCall {name: 'calculate'}) RETURN tc.id AS id, tc.parameters_json AS params"
        ).single()
        assert tc2_node_data is not None, "ToolCall 'calculate' not found in DB."
        if tc2_node_data:
            assert json.loads(tc2_node_data["params"]) == {"expression": "2+2"}
            rel2_check = db_sess_checker.run(
                "MATCH (:Event {id:$eid})-[:INVOKED_TOOL]->(:ToolCall {id:$tcid}) RETURN count(*) AS c",
                eid=evt.id, tcid=tc2_node_data["id"]
            ).single()
            assert rel2_check["c"] == 1, "INVOKED_TOOL relationship missing for calculate."
    sync_driver_checker.close()


@pytest.mark.asyncio
async def test_p7_shadow_app_state_creation_and_update(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests P7 AppState creation and updates via append_event."""
    app_name = "p7_app_test"
    user_id = "p7_user"
    sess = session_service.create_session(app_name=app_name, user_id=user_id)

    evt_actions1 = EventActions(state_delta={"app:theme": "dark", "app:feature_flag_x": True, "session_key": "val1"})
    evt1 = Event(author="user", content=types.Content(parts=[types.Part(text="Set app theme")]), actions=evt_actions1)
    session_service.append_event(sess, evt1)

    sync_driver_checker = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        app_state_node = db_sess_checker.run("MATCH (as:AppState {app_name: $app}) RETURN as.state_json AS state, as.version AS ver", app=app_name).single()
        assert app_state_node is not None
        app_state_data = json.loads(app_state_node["state"])
        assert app_state_data.get("app:theme") == "dark"
        assert app_state_data.get("app:feature_flag_x") is True
        assert "session_key" not in app_state_data
        original_app_version = app_state_node["ver"]

    evt_actions2 = EventActions(state_delta={"app:theme": "light", "app:new_setting": "enabled"})
    evt2 = Event(author="user", content=types.Content(parts=[types.Part(text="Update app theme")]), actions=evt_actions2)
    await asyncio.sleep(0.001)
    session_service.append_event(sess, evt2)

    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        app_state_node_updated = db_sess_checker.run("MATCH (as:AppState {app_name: $app}) RETURN as.state_json AS state, as.version AS ver", app=app_name).single()
        app_state_data_updated = json.loads(app_state_node_updated["state"])
        assert app_state_data_updated.get("app:theme") == "light"
        assert app_state_data_updated.get("app:feature_flag_x") is True
        assert app_state_data_updated.get("app:new_setting") == "enabled"
        assert app_state_node_updated["ver"] > original_app_version
    sync_driver_checker.close()


@pytest.mark.asyncio
async def test_p7_shadow_user_state_creation_and_update(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests P7 UserState creation and updates via append_event."""
    app_name = "p7_user_test_app"
    user_id = "p7_user1"
    sess = session_service.create_session(app_name=app_name, user_id=user_id)

    evt_actions1 = EventActions(state_delta={"user:preference": "notifications_on", "user:language": "en", "session_detail": "abc"})
    evt1 = Event(author="user", content=types.Content(parts=[types.Part(text="Set user preference")]), actions=evt_actions1)
    session_service.append_event(sess, evt1)

    sync_driver_checker = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        user_state_node = db_sess_checker.run("MATCH (us:UserState {app_name: $app, user_id: $uid}) RETURN us.state_json AS state, us.version AS ver", app=app_name, uid=user_id).single()
        assert user_state_node is not None
        user_state_data = json.loads(user_state_node["state"])
        assert user_state_data.get("user:preference") == "notifications_on"
        assert user_state_data.get("user:language") == "en"
        assert "session_detail" not in user_state_data
        original_user_version = user_state_node["ver"]

    evt_actions2 = EventActions(state_delta={"user:language": "fr", "user:last_login_ip": "1.2.3.4"})
    evt2 = Event(author="user", content=types.Content(parts=[types.Part(text="Update user language")]), actions=evt_actions2)
    await asyncio.sleep(0.001)
    session_service.append_event(sess, evt2)

    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        user_state_node_updated = db_sess_checker.run("MATCH (us:UserState {app_name: $app, user_id: $uid}) RETURN us.state_json AS state, us.version AS ver", app=app_name, uid=user_id).single()
        user_state_data_updated = json.loads(user_state_node_updated["state"])
        assert user_state_data_updated.get("user:preference") == "notifications_on"
        assert user_state_data_updated.get("user:language") == "fr"
        assert user_state_data_updated.get("user:last_login_ip") == "1.2.3.4"
        assert user_state_node_updated["ver"] > original_user_version
    sync_driver_checker.close()


def test_p7_create_session_merges_shadow_states(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests P7 create_session merging of pre-existing shadow states."""
    app_name = "p7_merge_app"
    user_id1 = "p7_merge_user1"

    sync_driver_modifier = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_modifier.session(database="neo4j") as db_sess_modifier:
        # Pre-populate AppState
        db_sess_modifier.run(
            "MERGE (as:AppState {app_name: $app}) SET as.state_json = $state, as.version = timestamp()",
            app=app_name, state=json.dumps({"app:global_setting": "active", "app:common_pref": "default_app"})
        )
        # Pre-populate UserState for user1
        db_sess_modifier.run(
            "MERGE (us:UserState {app_name: $app, user_id: $uid}) SET us.state_json = $state, us.version = timestamp()",
            app=app_name, uid=user_id1, state=json.dumps({"user:theme": "dark_user1", "user:custom_data": "user1_specific"})
        )
    sync_driver_modifier.close()

    sess1 = session_service.create_session(
        app_name=app_name, user_id=user_id1,
        state={"session_key": "val_s1", "app:common_pref": "session_override_app", "user:theme":"session_override_user"}
    )

    assert sess1.state.get("app:global_setting") == "active"
    assert sess1.state.get("app:common_pref") == "default_app" # Shadow wins
    assert sess1.state.get("user:theme") == "dark_user1" # Shadow wins
    assert sess1.state.get("user:custom_data") == "user1_specific"
    assert sess1.state.get("session_key") == "val_s1"


def test_p7_wrote_state_excludes_shadow_keys(session_service: Neo4jSessionService, neo4j_uri, neo4j_auth):
    """Tests P7 WROTE_STATE relationships exclude app:/user: keys."""
    app_name = "p7_wrote_app"
    user_id = "p7_wrote_user"
    sess = session_service.create_session(app_name=app_name, user_id=user_id, state={"session_data_point": "initial"})

    evt_actions = EventActions(state_delta={
        "app:config": "new_app_config",
        "user:profile_status": "active_user",
        "session_data_point": "xyz123",
        "another_session_key": "new_session_val"
    })
    evt = Event(author="user", content=types.Content(parts=[types.Part(text="Mixed state update")]), actions=evt_actions)
    session_service.append_event(sess, evt)

    sync_driver_checker = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with sync_driver_checker.session(database="neo4j") as db_sess_checker:
        # Check WROTE_STATE for session_data_point
        wrote_state_session_data = db_sess_checker.run(
            "MATCH (:Event {id:$eid})-[r:WROTE_STATE {key:'session_data_point'}]->(:Session {id:$sid}) RETURN r.fromValue_json, r.toValue_json",
            eid=evt.id, sid=sess.id
        ).single()
        assert wrote_state_session_data is not None, "WROTE_STATE for 'session_data_point' missing."
        if wrote_state_session_data:
             assert json.loads(wrote_state_session_data["fromValue_json"]) == "initial"
             assert json.loads(wrote_state_session_data["toValue_json"]) == "xyz123"

        # Check WROTE_STATE for another_session_key
        wrote_state_another_key = db_sess_checker.run(
            "MATCH (:Event {id:$eid})-[r:WROTE_STATE {key:'another_session_key'}]->(:Session {id:$sid}) RETURN r.fromValue_json, r.toValue_json",
            eid=evt.id, sid=sess.id
        ).single()
        assert wrote_state_another_key is not None, "WROTE_STATE for 'another_session_key' missing."
        if wrote_state_another_key:
            assert wrote_state_another_key["fromValue_json"] is None
            assert json.loads(wrote_state_another_key["toValue_json"]) == "new_session_val"

        # Check NO WROTE_STATE for app:config
        wrote_state_app_config = db_sess_checker.run(
            "MATCH (:Event {id:$eid})-[r:WROTE_STATE {key:'app:config'}]->(:Session {id:$sid}) RETURN count(r) AS c",
            eid=evt.id, sid=sess.id
        ).single()
        assert wrote_state_app_config["c"] == 0, "WROTE_STATE for 'app:config' should not exist."

        # Check NO WROTE_STATE for user:profile_status
        wrote_state_user_profile = db_sess_checker.run(
            "MATCH (:Event {id:$eid})-[r:WROTE_STATE {key:'user:profile_status'}]->(:Session {id:$sid}) RETURN count(r) AS c",
            eid=evt.id, sid=sess.id
        ).single()
        assert wrote_state_user_profile["c"] == 0, "WROTE_STATE for 'user:profile_status' should not exist."
    sync_driver_checker.close()
