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
    *   Alternatively, for newer Neo4j versions:
        ```cypher
        CREATE VECTOR INDEX memoryVectorIdx IF NOT EXISTS
        FOR (m:MemoryChunk) ON (m.embedding)
        OPTIONS {indexConfig:{`vector.dimensions`:1536, `vector.similarity_function`:'cosine'}};
        ```
        (Replace `1536` with your actual embedding dimension.)

## Installation

1.  **Prerequisites**:
    *   Python >= 3.8
    *   Access to a Neo4j database (version 4.4+ recommended, 5.x for vector index support).
    *   `uv` or `pip` for package installation.

2.  **Install Dependencies**:
    The package relies on `neo4j`, `google-adk`, and `google-genai`. These are listed in `pyproject.toml`.

    To install the package and its dependencies in editable mode (recommended for development):
    ```bash
    # From the root of this repository (e.g., /home/case/neo4j_services)
    uv pip install -e ./neo4j_adk_services
    # OR
    # python3 -m pip install -e ./neo4j_adk_services
    ```

## Configuration

### Neo4j Connection
Both services require Neo4j connection details:
*   `uri`: The Bolt URI of your Neo4j instance (e.g., `"bolt://localhost:7687"`).
*   `user`: The Neo4j username.
*   `password`: The Neo4j password.
*   `database`: (Optional) The specific Neo4j database name to use (defaults to the Neo4j default database).

### Neo4jMemoryService Specific Configuration
*   `embedding_function`: (Callable, optional) A function that takes a string of text and returns its vector embedding (list of floats). Required for vector search.
*   `vector_dimension`: (int, optional) The dimensionality of the vectors produced by `embedding_function`. Required if `embedding_function` is provided.
*   `similarity_top_k`: (int, optional, default: 5) The number of top similar results to retrieve in vector searches.
*   `vector_distance_threshold`: (float, optional) A threshold for vector similarity scores (e.g., cosine similarity). Results below this threshold might be filtered out.

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
#     dummy_event = ADKEvent(id="evt_mem_test", author="user", timestamp=time.time(), content=GenaiContent(parts=[GenaiPart(text="Test memory content about Neo4j.")]))
#     my_session = ADKSession(id="sess_mem_test", app_name="my_adk_app", user_id="user123", state={}, events=[dummy_event], last_update_time=time.time())


# Add a session's events to memory
# Ensure my_session is defined and has events
if 'my_session' in locals() and my_session.events:
    memory_service.add_session_to_memory(session=my_session)
    print(f"Session {my_session.id} processed for memory.")
else:
    print("Skipping add_session_to_memory as my_session is not defined or has no events.")


# Search memory
search_query = "Neo4j"
memory_results = memory_service.search_memory(
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


# Close the driver
memory_service.close()
```

## Running Tests

1.  **Prerequisites**:
    *   Docker installed and running.
    *   The `neo4j_adk_services` package and its `dev` dependencies installed (e.g., `uv pip install -e ".[dev]"`).

2.  **Start Neo4j Container**:
    A Neo4j instance must be running and accessible. The tests are configured to connect to `bolt://localhost:7687` with credentials `neo4j/test`. You can start a suitable container with:
    ```bash
    docker run -d --name neo4j-adk-test \
      --publish=7474:7474 --publish=7687:7687 \
      --env NEO4J_AUTH=neo4j/test \
      --env 'NEO4J_PLUGINS=["apoc"]' \
      neo4j:5 # Or a specific 5.x version like 2025.04.0 used in prompts
    ```
    Allow a few seconds for the container to initialize fully before running tests.

3.  **Execute Tests**:
    Navigate to the root of the workspace (`/home/case/neo4j_services`) and run:
    ```bash
    uv run python3 -m unittest discover -s neo4j_adk_services/tests -t .
    ```
    Or, if not using `uv run` and your virtual environment is activated:
    ```bash
    python3 -m unittest discover -s neo4j_adk_services/tests -t .
    ```

## Troubleshooting
*   **`TypeError: Can't instantiate abstract class Neo4jSessionService without an implementation for abstract method ...`**: Ensure all abstract methods from `BaseSessionService` are implemented in `Neo4jSessionService` with matching signatures (including keyword-only arguments where appropriate).
*   **`neo4j.exceptions.ServiceUnavailable: Couldn't connect to localhost:7687... Connection refused`**:
    *   Verify the Neo4j Docker container is running (`docker ps`).
    *   Ensure the port `7687` is correctly published.
    *   Allow sufficient time for Neo4j to start within the container before running applications or tests.
*   **`NameError: name '_helper_function' is not defined`**: Check that helper functions within service classes are correctly defined as methods (with `self`) and called with `self._helper_function(...)`.
*   **`KeyError` in test mocks**: The mock database logic in `tests/test_neo4j_services.py` might need updates if the Cypher queries or the structure of data returned by the services change significantly.

## Key Design Decisions & References
*   **Event Timeline with `NEXT`**: Inspired by temporal graph patterns for efficient ordered traversals. ([Neo4j Docs: Variable-length patterns](https://neo4j.com/docs/cypher-manual/current/patterns/variable-length-patterns/))
*   **State Lineage with `WROTE_STATE`**: Captures the history of state changes, allowing for audit trails and debugging. ([Stack Overflow: Neo4j strategy to keep history](https://stackoverflow.com/questions/22073512/neo4j-strategy-to-keep-history-of-node-changes))
*   **`ToolCall` Nodes**: Persists tool/function invocations (extracted from `google.genai.types.FunctionCall` objects in event content) as distinct `:ToolCall` nodes in Neo4j. This enables graph-based analysis of tool usage patterns, even though the ADK event structure itself has evolved.
*   **`DETACH DELETE`**: Used for `delete_session` to ensure cascading deletion of all related nodes and relationships. ([Neo4j Docs: Delete](https://neo4j.com/docs/cypher-manual/current/clauses/delete/#delete-delete-nodes-and-their-relationships))
*   **Constraints and Indexes**: Essential for data integrity and query performance. ([Neo4j Docs: Constraints](https://neo4j.com/docs/cypher-manual/current/constraints/syntax/), [Neo4j Docs: Full-text indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/), [Neo4j Docs: Vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/))
*   **APOC for JSON conversion in Cypher**: `apoc.convert.toJson()` is used in `WROTE_STATE` relationships to store complex `fromValue` and `toValue` as JSON strings. Ensure APOC plugin is installed in Neo4j.

This README provides a comprehensive guide to understanding, using, and developing the Neo4j services for Google ADK.