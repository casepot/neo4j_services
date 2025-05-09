import pytest
import asyncio
import time
from neo4j import GraphDatabase, AsyncGraphDatabase, exceptions as neo4j_exceptions
from testcontainers.neo4j import Neo4jContainer
import socket # For the new wait_port function

# New readiness probe: waits for TCP port to be open, no authentication.
def wait_port(host, port, timeout=30):
    """Waits for a TCP port to be open."""
    print(f"Waiting for port {port} on host {host} to open...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"Port {port} on host {host} is open.")
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Port {port} on host {host} never opened after {timeout}s.")


@pytest.fixture(scope="session")
def neo4j_container_instance():
    """Starts a Neo4j container and yields the container object."""
    print("Starting Neo4j container for session (TLS DISABLED)...")
    try:
        # Use password parameter in constructor and disable TLS
        with (
            Neo4jContainer("neo4j:5.26.6", password="letmein123") # Set password here
            .with_env("NEO4J_dbms_connector_bolt_tls__level", "DISABLED")
            .with_env("NEO4J_PLUGINS", '["apoc"]')
        ) as neo_container:
            # Wait for the Bolt port to be open using the new wait_port function
            host = neo_container.get_container_host_ip()
            # Neo4j Bolt port inside the container is 7687. Get the mapped port.
            mapped_port = neo_container.get_exposed_port(7687) # Changed to get_exposed_port
            wait_port(host, int(mapped_port))

            uri = neo_container.get_connection_url() # Should be plain bolt://
            print(f"Neo4j test-container is up (id={neo_container.get_wrapped_container().short_id}) — Bolt: {uri} (TLS DISABLED)")
            yield neo_container
    except Exception as e:
        print(f"Error starting Neo4j container: {e}")
        pytest.fail(f"Failed to start Neo4j container: {e}")
    # 'finally' block for stopping is handled by the 'with' statement context manager


@pytest.fixture(scope="session")
def neo4j_uri(neo4j_container_instance: Neo4jContainer):
    """Provides the Bolt URI for the running Neo4j container."""
    # testcontainers' Neo4jContainer.start() (called by 'with' context manager)
    # waits for the container to be ready. No need for _wait_for_neo4j.
    uri = neo4j_container_instance.get_connection_url()
    print(f"Neo4j container URI: {uri}")
    return uri

@pytest.fixture(scope="session")
def neo4j_auth():
    """Provides standard authentication tuple for Neo4j."""
    return ("neo4j", "letmein123")

@pytest.fixture(scope="session")
def neo4j_driver(neo4j_container_instance: Neo4jContainer):
    """Provides a ready-to-use Neo4j driver instance for the test session.
    This driver is configured by testcontainers to connect correctly (e.g. bolt+ssc, TrustAll).
    """
    driver = neo4j_container_instance.get_driver()
    yield driver
    driver.close()

@pytest.fixture(scope="session")
async def neo4j_async_uri(neo4j_container_instance: Neo4jContainer):
    """
    Provides the async Bolt URI, typically bolt+ssc for Neo4j 5.x.
    Note: neo4j_container_instance.get_driver() is preferred for obtaining a
    pre-configured driver. If constructing an async driver manually,
    ensure to use appropriate trust settings (e.g., TrustAll for self-signed certs).
    """
    # get_connection_url() returns "bolt://..."
    # For Neo4j 5.x with default encryption, we need "bolt+ssc://..."
    base_uri = neo4j_container_instance.get_connection_url()
    if base_uri.startswith("bolt://"):
        async_uri = base_uri.replace("bolt://", "bolt+ssc://")
    else:
        # If it's already bolt+ssc or something else, use as is, though this is unlikely
        # for get_connection_url() from the current testcontainers version.
        # Step 3: Ensure neo4j_async_uri returns plain bolt://
        # base_uri is already plain bolt:// from neo4j_uri fixture which calls get_connection_url()
        async_uri = base_uri # No replacement to bolt+ssc needed
        print(f"Neo4j async container URI: {async_uri} (TLS DISABLED)")
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