# Neo4j Services for Google ADK

This package provides Neo4j-backed implementations for the Session and Memory services of the Google Agent Development Kit (ADK). It enables developers to leverage a Neo4j graph database for persistent session storage, event tracking, and long-term memory recall with advanced graph-native features.

## Overview

The core components are:

*   **`Neo4jSessionService`**: A service that stores ADK session and event data in Neo4j. It extends `google.adk.sessions.BaseSessionService` and implements a rich graph model to capture not just session state but also the relationships between users, sessions, events, function calls (persisted as `:ToolCall` nodes), and state changes.
*   **`Neo4jMemoryService`**: A service for long-term memory storage and retrieval using Neo4j. It extends `google.adk.memory.BaseMemoryService` and supports both full-text and vector similarity searches over stored conversation events (represented as `MemoryChunk` nodes).
*   **`metric_helpers.py`** (Optional): Contains a `Metric` dataclass that can be used to model and store performance metrics (e.g., latency, token counts) associated with events.

This implementation focuses on a graph-native approach, making complex queries related to lineage, temporal analysis, and impact assessment more straightforward.

## Features

### Graph-Native Data Model
The services are designed to create a rich, interconnected graph in Neo4j, rather than just storing JSON blobs. This allows for powerful graph traversals and analytics.

**Key Entities and Relationships:**
*   **Nodes:**
    *   `App`: Represents an application using the ADK.
    *   `User`: Represents an end-user interacting with the application.
    *   `Session`: Represents a single conversation or interaction session.
    *   `Event`: Represents an individual event within a session (e.g., user message, agent response, function call). Note: Function calls are typically found within the `event.content`.
    *   `ToolCall`: Represents an invocation of an external tool/function during an event, persisted as a distinct node in Neo4j for analysis. Data is extracted from `google.genai.types.FunctionCall` objects found in the event content.
    *   `MemoryChunk`: Represents a piece of information (derived from an event) stored for long-term memory.
    *   `Metric` (Optional): Represents a performance or usage metric associated with an event.
    *   `AppState`: (P7) Represents global state for an application, shared across all users and sessions of that app. Identified by `app_name`.
    *   `UserState`: (P7) Represents global state for a specific user within an application, shared across all sessions of that user for that app. Identified by `app_name` and `user_id`.
*   **Relationships:**
    *   `(:User)-[:STARTED_SESSION]->(:Session)`: Links a user to the sessions they initiated.
    *   `(:Event)-[:OF_SESSION]->(:Session)`: Links an event to its parent session.
    *   `(:Event)-[:NEXT]->(:Event)`: Creates a chronological chain of events within a session for efficient timeline traversal.
    *   `(:Event)-[:WROTE_STATE {key, fromValue_json, toValue_json, timestamp}]->(:Session)`: Records changes to session-specific state (keys not prefixed with `app:` or `user:`), linking the event that caused the change to the session. It stores the key, previous value (as JSON), new value (as JSON), and timestamp of the change.
    *   `(:Event)-[:INVOKED_TOOL]->(:ToolCall)`: Links an event (specifically, the one containing the `FunctionCall` in its content) to the corresponding `:ToolCall` node representing that invocation.
    *   `(:Metric)-[:FOR_EVENT]->(:Event)` (Optional): Links a metric to the event it pertains to.
    *   `(:MemoryChunk)-[:SIMILAR {score}]->(:MemoryChunk)` (Conceptual): Represents semantic similarity between memory chunks. This relationship is intended to be populated offline, for example, using Neo4j Graph Data Science (GDS) k-NN algorithms.

### Neo4jSessionService
*   **Session Management**: Creates and retrieves sessions, linking them to `App` and `User` nodes. When a session is created, it merges in the latest state from corresponding `AppState` and `UserState` shadow nodes.
*   **Event Persistence**: Appends events to sessions, automatically creating `NEXT` relationships to maintain order.
*   **State Change Tracking**:
    *   For session-specific keys (not prefixed with `app:` or `user:`) in `event.actions.state_delta`, a `WROTE_STATE` relationship is created from the `Event` to the `Session`, capturing the key, old value, new value, and timestamp.
    *   **Shadow State Updates (P7)**: For keys prefixed with `app:` or `user:` in `event.actions.state_delta`, the service updates corresponding `AppState` or `UserState` nodes. These shadow nodes store their state as a JSON string and have a `version` timestamp. Updates use `apoc.map.merge` for handling concurrent writes (last writer wins for a given key within the shadow node).
*   **Tool Call Tracking**: If an event's content contains `google.genai.types.FunctionCall` objects (retrieved via `event.get_function_calls()`), corresponding `:ToolCall` nodes are created in Neo4j with relevant details (name, arguments as JSON). These nodes are linked from the event via `INVOKED_TOOL` relationships.
*   **Ephemeral State Handling**: Keys prefixed with `"temp:"` in `state_delta` are not persisted, aligning with ADK conventions.
*   **Optimistic Locking**: When appending events, the service checks `Session.last_update_time` (converted to milliseconds) against the database version to prevent stale writes. If a mismatch occurs, a `StaleSessionError` is raised.

### Neo4jMemoryService
*   **Memory Ingestion**: Stores relevant information from session events as distinct `MemoryChunk` nodes. Each `MemoryChunk` typically contains the text content, author, timestamp, the ID of the original event (`eid`), and the `session_id` it belongs to. It also stores `app_name` and `user_id` for partitioned search.
*   **Full-Text Search**: Implements keyword-based search over the `text` property of `MemoryChunk` nodes using Neo4j's full-text indexing.
*   **Vector Search (Semantic Similarity)**:
    *   Supports storing embedding vectors (e.g., from OpenAI, Gemini) on `MemoryChunk` nodes.
    *   Utilizes Neo4j's vector indexes for approximate nearest neighbor (ANN) search to find semantically similar memories.
    *   Requires an `embedding_function` and `vector_dimension` to be provided during instantiation.
*   **Dual Search Strategy**: Can perform vector search if configured, and falls back to or combines with full-text search.

## Schema Details

### Node Labels and Key Properties
*   **`App`**:
    *   `name`: (string, unique for app identification)
*   **`User`**:
    *   `id`: (string, unique for user identification)
*   **`Session`**:
    *   `id`: (string, unique session identifier)
    *   `app_name`: (string)
    *   `user_id`: (string)
    *   `state_json`: (string, JSON representation of the session state)
    *   `last_update_time`: (integer, millisecond Unix epoch timestamp for optimistic locking in DB; float seconds in Python `Session` object)
*   **`Event`**:
    *   `id`: (string, unique event identifier)
    *   `author`: (string, e.g., "user", "agent", "tool")
    *   `timestamp`: (float)
    *   `invocation_id`: (string, optional)
    *   `content_json`: (string, JSON representation of `event.content`)
    *   `actions_json`: (string, JSON representation of `event.actions`)
    *   `text`: (string, textual summary of content for search)
*   **`ToolCall`** (Neo4j Node representing a function/tool invocation):
    *   `id`: (string, unique identifier generated by the service for the node)
    *   `name`: (string, name of the function/tool, from `FunctionCall.name`)
    *   `parameters_json`: (string, JSON representation of function arguments, from `FunctionCall.args`)
    *   `latency_ms`: (integer, optional - *Note: Not directly available from `FunctionCall`*)
    *   `status`: (string, optional - *Note: Not directly available from `FunctionCall`*)
    *   `error_msg`: (string, optional - *Note: Not directly available from `FunctionCall`*)
*   **`MemoryChunk`**:
    *   `eid`: (string, ID of the original event this chunk is derived from)
    *   `text`: (string, textual content for search and recall)
    *   `author`: (string)
    *   `ts`: (float, timestamp of the original event)
    *   `app_name`: (string, for partitioning)
    *   `user_id`: (string, for partitioning)
    *   `session_id`: (string, ID of the session this chunk belongs to)
    *   `embedding`: (list of floats, optional vector embedding)
*   **`Metric`** (Optional):
    *   `id`: (string, unique metric identifier)
    *   `name`: (string, e.g., "token_count", "latency")
    *   `value`: (float)
    *   `unit`: (string, optional, e.g., "ms", "tokens")
    *   `ts`: (float, timestamp)
*   **`AppState`** (P7):
    *   `app_name`: (string, unique identifier for the application, e.g., "my_adk_app")
    *   `state_json`: (string, JSON representation of app-specific global state)
    *   `version`: (integer, millisecond Unix epoch timestamp, updated on modification)
*   **`UserState`** (P7):
    *   `app_name`: (string, application identifier)
    *   `user_id`: (string, user identifier)
    *   `state_json`: (string, JSON representation of user-specific global state for that app)
    *   `version`: (integer, millisecond Unix epoch timestamp, updated on modification)

### Constraints and Indexes
The services attempt to create the following constraints and indexes upon initialization if they don't already exist. These operations are designed to be idempotent (e.g., using `IF NOT EXISTS` or by handling `ClientError` exceptions for procedure-based index creation), making service restarts against an existing database safer. Ensure the Neo4j user has permissions to create them.

*   **Constraints**:
    *   `CREATE CONSTRAINT session_unique IF NOT EXISTS FOR (s:Session) REQUIRE (s.app_name, s.user_id, s.id) IS UNIQUE;` (Handled by `Neo4jSessionService`)
    *   `CREATE CONSTRAINT tool_call_id_unique IF NOT EXISTS FOR (t:ToolCall) REQUIRE t.id IS UNIQUE;` (Handled by `Neo4jSessionService`)
    *   `CREATE CONSTRAINT app_state_unique IF NOT EXISTS FOR (a:AppState) REQUIRE a.app_name IS UNIQUE;` (P7, Handled by `Neo4jSessionService`)
    *   `CREATE CONSTRAINT user_state_unique IF NOT EXISTS FOR (u:UserState) REQUIRE (u.app_name, u.user_id) IS UNIQUE;` (P7, Handled by `Neo4jSessionService`)
    *   (Conceptual, if `Metric` nodes are used) `CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE;`
*   **Full-Text Index** (for `Neo4jMemoryService`):
    *   `CALL db.index.fulltext.createNodeIndex('MemoryTextIndex', ['MemoryChunk'], ['text'])` (Handled by `Neo4jMemoryService`)
*   **Vector Index** (for `Neo4jMemoryService`, if `vector_dimension` is provided):
    *   `CALL db.index.vector.createNodeIndex('MemoryVectorIndex', 'MemoryChunk', 'embedding', $dim, 'cosine')` (Handled by `Neo4jMemoryService`)
    *   Alternatively, for newer Neo4j versions (5.7+):
        ```cypher
        CREATE VECTOR INDEX MemoryVectorIndex IF NOT EXISTS
        FOR (m:MemoryChunk) ON (m.embedding)
        OPTIONS {indexConfig:{`vector.dimensions`:1536, `vector.similarity_function`:'cosine'}};
        ```
        (Replace `1536` with your actual embedding dimension.)

## Installation

1.  **Prerequisites**:
    *   Python >= 3.8
    *   Access to a Neo4j database (version 5.x recommended, e.g., `5.26.6` for tests).
    *   `uv` or `pip` for package installation.

2.  **Install Dependencies**:
    The package relies on `neo4j`, `google-adk`, `google-genai`, and `testcontainers` (for development).
    Key versions (see `pyproject.toml` for exact pinning):
    *   `neo4j>=5.28,<6`
    *   `testcontainers[neo4j]>=4.10,<5`

    To install the package and its dependencies in editable mode (recommended for development):
    ```bash
    # From the root of this repository (e.g., /home/case/neo4j_services)
    # Ensure to quote the argument if using zsh to prevent globbing issues:
    uv pip install -e "./neo4j_adk_services[dev]"
    # OR
    # python3 -m pip install -e "./neo4j_adk_services[dev]"
    ```

## Configuration

### Neo4j Connection
Both services require Neo4j connection details:
*   `uri`: The Bolt URI of your Neo4j instance (e.g., `"bolt://localhost:7687"`).
*   `user`: The Neo4j username.
*   `password`: The Neo4j password.
*   `database`: (Optional) The specific Neo4j database name to use (defaults to the Neo4j default database).
*   **Note on Test Environment**: The project's test environment (configured in `tests/conftest.py`) starts the Neo4j container with TLS disabled (`NEO4J_dbms_connector_bolt_tls__level="DISABLED"`) and uses plain `bolt://` URIs for all driver connections. For production or other environments where TLS is enabled on the Neo4j server, ensure your URI (e.g., `bolt+s://` or `bolt+ssc://`) and driver settings (e.g., `encrypted=True`, certificate trust) match the server's requirements.

### Neo4jMemoryService Specific Configuration
*   `embedding_function`: (Callable, optional) A function that takes a string of text and returns its vector embedding (list of floats). Required for vector search.
*   `vector_dimension`: (int, optional) The dimensionality of the vectors produced by `embedding_function`. Required if `embedding_function` is provided.
*   `similarity_top_k`: (int, optional, default: 5) The number of top similar results to retrieve in vector searches.
*   `vector_distance_threshold`: (float, optional) A threshold for vector similarity scores (e.g., cosine similarity). Results below this threshold might be filtered out.
*   `max_concurrent_requests`: (int, optional, default: 10) Maximum concurrent requests to the database, managed by an internal semaphore.

### Async Upgrade Notice (Neo4jMemoryService)
**Breaking Change**: The `Neo4jMemoryService` has been updated to use an asynchronous Neo4j driver for improved performance.
*   `add_session_to_memory` is now an `async def` method.
*   `search_memory` is now an `async def` method.
*   A synchronous wrapper `search_memory_sync` is provided for convenience during transition:
    ```python
    # If you need to call search_memory from synchronous code:
    results = memory_service.search_memory_sync(app_name="app", user_id="user", query="search term")
    ```
    For new asynchronous code, use `await memory_service.search_memory(...)`.

## Usage

### Neo4jSessionService

```python
from neo4j_adk_services.src import Neo4jSessionService
from google.adk.sessions import Session
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part # If using these types

# Initialize the service
session_service = Neo4jSessionService(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)

# Create a new session
my_session = session_service.create_session(
    app_name="my_adk_app",
    user_id="user123",
    state={"initial_key": "initial_value"}
)
print(f"Session created: {my_session.id}")

# Append an event
event_content = Content(parts=[Part(text="Hello, agent!")])
event_actions = EventActions(state_delta={"mood": "curious", "temp:source": "user_input"})
my_event = Event(author="user", content=event_content, actions=event_actions)

session_service.append_event(session=my_session, event=my_event)
print(f"Event appended. Session state: {my_session.state}")

# Get a session
retrieved_session = session_service.get_session(
    app_name="my_adk_app",
    user_id="user123",
    session_id=my_session.id
)
if retrieved_session:
    print(f"Retrieved session with {len(retrieved_session.events)} events.")

# List sessions for a user
user_sessions = session_service.list_sessions(
    app_name="my_adk_app",
    user_id="user123"
)
# list_sessions returns a ListSessionsResponse object
print(f"User user123 has {len(user_sessions.sessions)} sessions.")

# If append_event is called with a session object whose last_update_time
# does not match the one in the database (e.g., due to a concurrent update),
# a StaleSessionError will be raised. Callers should catch this and can
# retry the operation after re-fetching the session.

# ... (other operations like list_events, delete_session)

# Close the driver when application shuts down
session_service.close()
```

### Neo4jMemoryService

```python
from neo4j_adk_services.src import Neo4jMemoryService
# Assuming you have a session object from Neo4jSessionService or similar
# from neo4j_adk_services.src import Neo4jSessionService (as above)

# Example embedding function (replace with your actual model)
def get_dummy_embedding(text: str) -> list[float]:
    # In a real scenario, use a sentence transformer, OpenAI API, etc.
    # Dimension must match vector_dimension
    return [hash(char) / 10e10 for char in text[:128]] # Dummy 128-dim embedding

memory_service = Neo4jMemoryService(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password",
    embedding_function=get_dummy_embedding, # Optional
    vector_dimension=128 # Required if embedding_function is provided
)

# Assume 'my_session' is a populated Session object
# (e.g., retrieved via session_service.get_session or after appending events)
# For this example, let's create a dummy one if not available:
# if 'my_session' not in locals():
#     from google.adk.sessions import Session as ADKSession
#     from google.adk.events import Event as ADKEvent
#     from google.genai.types import Content as GenaiContent, Part as GenaiPart
#     import time
#     dummy_event = ADKEvent(id="evt_mem_test", author="user", timestamp=time.time(), content=GenaiContent(parts=[GenaiPart(text="Test memory content about Neo4j.")]))
#     my_session = ADKSession(id="sess_mem_test", app_name="my_adk_app", user_id="user123", state={}, events=[dummy_event], last_update_time=time.time())


# Add a session's events to memory
# Ensure my_session is defined and has events
if 'my_session' in locals() and hasattr(my_session, 'events') and my_session.events:
    # Neo4jMemoryService.add_session_to_memory is async
    import asyncio
    asyncio.run(memory_service.add_session_to_memory(session=my_session))
    print(f"Session {my_session.id} processed for memory.")
else:
    print("Skipping add_session_to_memory as my_session is not defined or has no events.")


# Search memory (async)
search_query = "Neo4j"
# memory_results = asyncio.run(memory_service.search_memory( # if running from sync context
memory_results = memory_service.search_memory_sync( # using the sync wrapper
    app_name="my_adk_app",
    user_id="user123",
    query=search_query
)

# The service returns a SearchMemoryResponse object
if hasattr(memory_results, "memories"): # memory_results is a SearchMemoryResponse object
     for mem_item in memory_results.memories: # mem_item is a MemoryResult object
        print(f"  Session ID: {mem_item.session_id}")
        # MemoryResult contains a list of Event objects
        for event_detail in mem_item.events:
            event_text = ""
            if event_detail.content and hasattr(event_detail.content, 'parts'):
                event_text = " ".join([part.text for part in event_detail.content.parts if hasattr(part, 'text')])
            print(f"    Event ({event_detail.id} by {event_detail.author}): {event_text}")


# Close the driver (async)
# asyncio.run(memory_service.close()) # if running from sync context
# For simplicity, if the main program is sync, and this is the end, direct call might be okay
# but proper async teardown is better. If memory_service was used in an async context,
# await memory_service.close() would be used there.
# Since search_memory_sync was used, we assume a sync context for close as well for this example.
# However, the .close() method itself is async.
# A simple way if you are at the very end of a script:
loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(memory_service.close())
except RuntimeError as e:
    if "Cannot run an event loop while another is running" in str(e):
        # If an outer loop is already running (e.g. in Jupyter), this might be tricky.
        # For simple scripts, creating a new loop if none is running is an option.
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(memory_service.close())
    else:
        raise
```

## Running Tests

1.  **Prerequisites**:
    *   Docker installed and running (for integration tests).
    *   The `neo4j_adk_services` package and its `dev` dependencies installed (e.g., `uv pip install -e "./neo4j_adk_services[dev]"`). This will include `pytest`, `pytest-asyncio`, and `testcontainers` (version `4.10.x` or as specified in `pyproject.toml`).

2.  **Execute Tests**:
    The integration tests use `Testcontainers` to automatically spin up a Neo4j Docker container (version `5.26.6` as configured in `tests/conftest.py`). The container is configured with the APOC plugin enabled (via `NEO4J_PLUGINS`) and **TLS disabled** (via `NEO4J_dbms_connector_bolt_tls__level="DISABLED"`). All test connections use plain `bolt://` URIs. There is no need to manually start a Neo4j container for running the tests.

    Navigate to the root of the workspace (`/home/case/neo4j_services`) and run:
    ```bash
    pytest neo4j_adk_services/tests -v
    ```
    Or, using `uv`:
    ```bash
    uv run pytest neo4j_adk_services/tests -v
    ```
    The tests will:
    *   Start a Neo4j container.
    *   Run all unit and integration tests, including those for asynchronous methods and vector search capabilities against the live container.
    *   Automatically stop the Neo4j container after tests complete.

## Troubleshooting

This section summarizes key connection and authentication issues encountered during development and testing, and their resolutions, primarily focusing on the `testcontainers` setup.

*   **Initial Connection Failures (`AuthError`, `SSLEOFError`, `ConfigurationError`, `AuthenticationRateLimit`)**:
    A series of connection problems were encountered when setting up the test environment with `testcontainers` and Neo4j 5.x. These included:
    *   `neo4j.exceptions.AuthError`: Incorrect password or user not yet available.
    *   `SSLEOFError`: SSL handshake failures, often due to a mismatch between client and server TLS expectations or certificate issues.
    *   `neo4j.exceptions.ConfigurationError`: Incorrect combination of URI schemes (e.g., `bolt+ssc://`) and explicit driver encryption parameters (e.g., `encrypted=True`). The driver infers settings from schemes like `bolt+ssc://`.
    *   `neo4j.exceptions.AuthError: {code: Neo.ClientError.Security.AuthenticationRateLimit}`: Triggered by too many failed login attempts, often because readiness probes were trying to authenticate before the Neo4j container's auth subsystem was fully initialized.

*   **Resolution Path for Test Environment**:
    The most stable configuration for the test environment was achieved by:
    1.  **Disabling TLS in the Neo4j Container**:
        In `tests/conftest.py`, the `Neo4jContainer` is initialized with the environment variable `NEO4J_dbms_connector_bolt_tls__level="DISABLED"`.
        ```python
        Neo4jContainer("neo4j:5.26.6", password="letmein123")
            .with_env("NEO4J_dbms_connector_bolt_tls__level", "DISABLED")
            # ... other .with_env() calls
        ```
    2.  **Using Plain `bolt://` URIs Everywhere**:
        All Neo4j driver initializations (in `tests/conftest.py` fixtures like `neo4j_uri`, `neo4j_async_uri`; in service constructors `Neo4jSessionService`, `Neo4jMemoryService`; and in the `clear_db` test fixture) now consistently use the plain `bolt://` URI returned by `neo_container.get_connection_url()`. No `bolt+ssc://` conversions are made, and no explicit `encrypted=True` or `trust_all_certificates=True` flags are passed to the driver, as TLS is disabled at the server level.
    3.  **Reliable Password Setting**:
        The initial password for the test container is set using the `password` parameter in the `Neo4jContainer` constructor: `Neo4jContainer("neo4j:5.26.6", password="letmein123")`.
    4.  **Non-Authenticated Readiness Probe**:
        The readiness probe in `tests/conftest.py` (`wait_port` function) now checks for TCP socket availability on the mapped Bolt port without attempting an authenticated Neo4j connection. This prevents early login failures and the `AuthenticationRateLimit` error.
        ```python
        # In tests/conftest.py within neo4j_container_instance fixture:
        host = neo_container.get_container_host_ip()
        mapped_port = neo_container.get_exposed_port(7687) # Ensure correct method name
        wait_port(host, int(mapped_port))
        ```
    5.  **Correct Testcontainers API Usage**:
        Ensured correct `testcontainers` API methods were used (e.g., `get_exposed_port()` instead of `get_mapped_port()`).

*   **General `TypeError: Can't instantiate abstract class ...`**: Ensure all abstract methods from base classes (`BaseSessionService`, `BaseMemoryService`) are implemented with matching signatures in the derived Neo4j service classes.
*   **`ImportError` or `AttributeError` related to `testcontainers`**: Verify the installed version of `testcontainers-python` and consult its documentation for correct API usage, as methods can change between versions (e.g., `Neo4jLabsPlugin` not in Python version, `get_container_name` vs `get_wrapped_container().short_id`).
*   **Shell Errors (e.g., `zsh: no matches found` for `pip install`)**: Quote arguments containing special characters like square brackets: `uv pip install -e "./neo4j_adk_services[dev]"`.

This setup ensures that the test environment consistently uses non-encrypted connections, simplifying the connection logic and avoiding SSL-related complexities. For production, a TLS-enabled setup would be recommended, requiring careful alignment of server configuration, URIs (e.g., `bolt+s://` or `bolt+ssc://`), and driver trust settings.

## Key Design Decisions & References
*   **Event Timeline with `NEXT`**: Inspired by temporal graph patterns for efficient ordered traversals. ([Neo4j Docs: Variable-length patterns](https://neo4j.com/docs/cypher-manual/current/patterns/variable-length-patterns/))
*   **State Lineage with `WROTE_STATE`**: Captures the history of state changes, allowing for audit trails and debugging. ([Stack Overflow: Neo4j strategy to keep history](https://stackoverflow.com/questions/22073512/neo4j-strategy-to-keep-history-of-node-changes))
*   **`ToolCall` Nodes**: Persists tool/function invocations (extracted from `google.genai.types.FunctionCall` objects in event content) as distinct `:ToolCall` nodes in Neo4j. This enables graph-based analysis of tool usage patterns.
*   **`DETACH DELETE`**: Used for `delete_session` to ensure cascading deletion of all related nodes and relationships. ([Neo4j Docs: Delete](https://neo4j.com/docs/cypher-manual/current/clauses/delete/#delete-delete-nodes-and-their-relationships))
*   **Constraints and Indexes**: Essential for data integrity and query performance. ([Neo4j Docs: Constraints](https://neo4j.com/docs/cypher-manual/current/constraints/syntax/), [Neo4j Docs: Full-text indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/), [Neo4j Docs: Vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/))
*   **APOC for JSON conversion in Cypher**: `apoc.convert.toJson()` is used in `WROTE_STATE` relationships to store complex `fromValue` and `toValue` as JSON strings. Ensure APOC plugin is installed in Neo4j (handled by Testcontainers setup via `NEO4J_PLUGINS` env var).
*   **Testcontainers Python API**: For container management in tests, refer to the [Testcontainers Python Documentation](https://testcontainers-python.readthedocs.io/).

This README provides a comprehensive guide to understanding, using, and developing the Neo4j services for Google ADK.