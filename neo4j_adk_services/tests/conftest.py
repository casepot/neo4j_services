import pytest
import asyncio
import time
from neo4j import GraphDatabase, AsyncGraphDatabase, exceptions as neo4j_exceptions
from testcontainers.neo4j import Neo4jContainer

# Helper function to wait for Neo4j to be ready
def _wait_for_neo4j(uri: str, user="neo4j", password="test", retries=20, delay=5):
    """Polls Neo4j until it's responsive or retries are exhausted."""
    sync_driver = None
    for i in range(retries):
        try:
            sync_driver = GraphDatabase.driver(uri, auth=(user, password))
            with sync_driver.session() as session:
                session.run("RETURN 1")
            print(f"Neo4j at {uri} is responsive.")
            return True
        except neo4j_exceptions.ServiceUnavailable:
            print(f"Neo4j not yet available at {uri}. Retrying in {delay}s... ({i+1}/{retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"An unexpected error occurred while waiting for Neo4j: {e}")
            time.sleep(delay) # Still wait before retrying
        finally:
            if sync_driver:
                sync_driver.close()
    raise ConnectionError(f"Neo4j container did not become responsive at {uri} after {retries} retries.")

@pytest.fixture(scope="session")
def neo4j_container_instance():
    """Starts a Neo4j container and yields the container object."""
    print("Starting Neo4j container for session...")
    # It's good practice to ensure the image is explicitly set if not default.
    try:
        with Neo4jContainer("neo4j:5.26.6") \
                .with_env("NEO4J_AUTH", "neo4j/test") \
                .with_env("NEO4J_PLUGINS", '["apoc"]') as neo_container:
            # v4+: use get_wrapped_container() to access the underlying Docker container object
            wrapped_container = neo_container.get_wrapped_container()
            print(f"Neo4j test-container is up (id={wrapped_container.short_id}) — Bolt: "
                  f"{neo_container.get_connection_url()}")
            yield neo_container
    except Exception as e:
        print(f"Error starting Neo4j container: {e}")
        pytest.fail(f"Failed to start Neo4j container: {e}")
    finally:
        print("Neo4j container for session is stopping/stopped.")


@pytest.fixture(scope="session")
def neo4j_uri(neo4j_container_instance: Neo4jContainer):
    """Provides the Bolt URI for the running Neo4j container and waits for it to be ready."""
    uri = neo4j_container_instance.get_connection_url()
    print(f"Neo4j container URI: {uri}")
    # Wait for the Neo4j instance inside the container to be ready
    _wait_for_neo4j(uri, user="neo4j", password="test")
    return uri

@pytest.fixture(scope="session")
def neo4j_auth():
    """Provides standard authentication tuple for Neo4j."""
    return ("neo4j", "test")

@pytest.fixture(scope="session")
async def neo4j_async_uri(neo4j_uri: str):
    """Provides the async Bolt URI for the running Neo4j container."""
    # Async driver typically uses bolt+ssc or bolt+s for encrypted connections
    # or bolt for unencrypted. get_connection_url() returns bolt://
    # For async, ensure the scheme is appropriate if encryption is involved.
    # The issue suggests replacing "bolt://" with "bolt+ssc://".
    # This implies the container is expected to support SSC.
    # For local test containers, often 'bolt://' is fine for async too if not enforcing encryption.
    # Let's stick to the issue's guidance for SSC.
    async_uri = neo4j_uri.replace("bolt://", "bolt+ssc://")
    print(f"Neo4j async container URI: {async_uri}")
    return async_uri

@pytest.fixture(scope="session")
def event_loop():
    """Ensure an event loop is available for session-scoped async fixtures."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    # loop.close() # Closing the loop can cause issues if other tests need it.
                 # Pytest-asyncio usually handles loop management.

# Example of how a service could be instantiated for tests using the fixture
# This would typically be in the test file itself or a more specific conftest.
# from neo4j_adk_services.src import Neo4jSessionService, Neo4jMemoryService
#
# @pytest.fixture(scope="function") # or "module"
# def session_service_real_db(neo4j_uri, neo4j_auth):
#     service = Neo4jSessionService(uri=neo4j_uri, user=neo4j_auth[0], password=neo4j_auth[1], database="neo4j")
#     # Optional: clear database before each test function if scope is "function"
#     # with service._driver.session(database="neo4j") as s:
#     #     s.run("MATCH (n) DETACH DELETE n")
#     yield service
#     service.close()
#
# @pytest_asyncio.fixture(scope="function") # or "module"
# async def memory_service_real_db(neo4j_async_uri, neo4j_auth):
#     # Dummy embedding function for tests
#     def get_dummy_embedding(text: str) -> list[float]:
#         return [hash(char) / 1e12 for char in text[:128]] # Ensure dimension matches
#
#     service = Neo4jMemoryService(
#         uri=neo4j_async_uri,
#         user=neo4j_auth[0],
#         password=neo4j_auth[1],
#         database="neo4j",
#         embedding_function=get_dummy_embedding,
#         vector_dimension=128 # Example dimension
#     )
#     # Optional: clear database or specific nodes
#     # async with service._driver.session(database="neo4j") as s:
#     #     await s.run("MATCH (m:MemoryChunk) DETACH DELETE m")
#     yield service
#     await service.close()