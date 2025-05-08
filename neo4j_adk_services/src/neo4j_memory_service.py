import neo4j # Added for exception handling
from neo4j import AsyncGraphDatabase # Changed from GraphDatabase
import asyncio # For async operations and semaphore

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
    MemoryResult,
)
from google.adk.sessions import Session # Added for type hinting
from google.adk.events import Event # For constructing MemoryResult
from google.genai.types import Content, Part # For constructing Event content

class Neo4jMemoryService(BaseMemoryService):
    """A MemoryService implementation backed by Neo4j, supporting full-text and vector search."""
    def __init__(self, uri: str, user: str, password: str, database: str = None,
                 embedding_function: callable = None, vector_dimension: int = None, similarity_top_k: int = 5,
                 vector_distance_threshold: float = None, max_concurrent_requests: int = 10):
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password)) # Changed to AsyncGraphDatabase
        self._database = database
        self._embedding_fn = embedding_function
        self._vector_dim = vector_dimension # Configured dimension for index creation
        self._top_k = similarity_top_k or 5
        self._vector_threshold = vector_distance_threshold
        
        self._indexes_ensured = False
        self._index_creation_lock = asyncio.Lock()
        self._mem_idx_dim_cache = None # Cache for actual vector index dimension from DB
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Index creation is deferred to _ensure_indexes, called by async methods.
        super().__init__()

    async def _ensure_indexes(self):
        """Ensures that necessary database indexes are created. Idempotent."""
        async with self._index_creation_lock:
            if self._indexes_ensured:
                return

            # Full-text index
            try:
                await self._execute_write(
                    "CALL db.index.fulltext.createNodeIndex('MemoryTextIndex', ['MemoryChunk'], ['text'])"
                )
                print("Successfully created/ensured full-text index 'MemoryTextIndex'.")
            except neo4j.exceptions.ClientError as e:
                if "An equivalent index already exists" not in str(e) and "already exists" not in str(e):
                    print(f"Failed to create full-text index 'MemoryTextIndex': {e}")
                    # raise # Or handle more gracefully
                else:
                    print("Full-text index 'MemoryTextIndex' already exists.")
            
            # Vector index
            if self._vector_dim:
                index_name = 'MemoryVectorIndex'
                created_via_procedure = False
                # Try procedure first (older Neo4j 5.x)
                try:
                    await self._execute_write(
                        f"CALL db.index.vector.createNodeIndex('{index_name}', 'MemoryChunk', 'embedding', $dim, 'cosine')",
                        {"dim": int(self._vector_dim)}
                    )
                    print(f"Successfully created/ensured vector index '{index_name}' using procedure.")
                    created_via_procedure = True
                except neo4j.exceptions.ClientError as e_proc:
                    if "An equivalent index already exists" in str(e_proc) or "already exists" in str(e_proc):
                        print(f"Vector index '{index_name}' (procedure) already exists.")
                        created_via_procedure = True # Assume it exists with correct config
                    elif "Unknown procedure" in str(e_proc) or "No procedure" in str(e_proc):
                        # Procedure might not exist on newer versions, try DDL
                        print(f"Vector index procedure for '{index_name}' not found, trying DDL.")
                        try:
                            await self._execute_write(
                                f"""CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                                FOR (m:MemoryChunk) ON (m.embedding)
                                OPTIONS {{indexConfig:{{`vector.dimensions`:{int(self._vector_dim)}, `vector.similarity_function`:'cosine'}}}}"""
                            )
                            print(f"Successfully created/ensured vector index '{index_name}' using DDL.")
                        except neo4j.exceptions.ClientError as e_ddl:
                            if "An equivalent index already exists" not in str(e_ddl) and "already exists" not in str(e_ddl):
                                print(f"Failed to create vector index '{index_name}' using DDL after procedure failed: {e_ddl}")
                            else:
                                print(f"Vector index '{index_name}' (DDL) already exists or was concurrently created.")
                    else:
                        # Some other error with the procedure call
                        print(f"Failed to create vector index '{index_name}' using procedure: {e_proc}")
                
                # After attempting creation, fetch and cache the dimension from DB
                # This also serves as a check that the index exists with some dimension
                if self._mem_idx_dim_cache is None: # Only fetch if not already cached
                    dim_from_db = await self._get_vector_index_dimension(index_name)
                    if dim_from_db is not None:
                        if dim_from_db != int(self._vector_dim):
                            # This is a critical mismatch if index was pre-existing with different dimension
                            raise ValueError(f"Configured vector_dimension ({self._vector_dim}) "
                                             f"does not match existing index dimension ({dim_from_db}) for '{index_name}'. "
                                             "Please check Neo4j server configuration or service init parameters.")
                        self._mem_idx_dim_cache = dim_from_db
                        print(f"Cached dimension for '{index_name}': {self._mem_idx_dim_cache}")
                    elif self._vector_dim: # If we intended to create it and still can't get dim
                         print(f"Warning: Could not verify dimension for vector index '{index_name}' from database after creation attempt. "
                               f"Will rely on configured dimension: {self._vector_dim}")
                         self._mem_idx_dim_cache = int(self._vector_dim) # Fallback to configured, but this is risky

            self._indexes_ensured = True

    async def _get_vector_index_dimension(self, index_name: str) -> int | None:
        """Fetches the dimension of a given vector index from the database."""
        query = """
        CALL db.indexes() YIELD name, type, options
        WHERE name = $idx_name AND type = 'VECTOR'
        RETURN toInteger(options['vector.dimensions']) AS dim
        """
        # Note: For Neo4j 5.18+ options might be a map e.g. options.indexConfig.`vector.dimensions`
        # The provided query `options['vector.dimensions']` is for older syntax or specific setups.
        # Let's try to make it more robust or stick to the issue's specified query.
        # The issue's query: CALL db.indexes() YIELD name, entityType, properties, type, options WHERE name = $idx AND type = 'VECTOR' RETURN toInteger(options['vector.dimensions']) AS dim
        # This seems fine.
        params = {"idx_name": index_name}
        try:
            result = await self._execute_read(query, params)
            if result and result[0] and "dim" in result[0] and result[0]["dim"] is not None:
                return int(result[0]["dim"])
            else: # Index not found or dimension property missing
                # Try alternative for newer Neo4j versions if the above fails (e.g. 5.13+)
                # This is for `SHOW VECTOR INDEXES` or specific `options` structure
                alt_query = """
                SHOW VECTOR INDEXES YIELD name, options
                WHERE name = $idx_name
                RETURN options.indexConfig.`vector.dimensions` AS dim
                """
                try:
                    alt_result = await self._execute_read(alt_query, params)
                    if alt_result and alt_result[0] and "dim" in alt_result[0] and alt_result[0]["dim"] is not None:
                        return int(alt_result[0]["dim"])
                except neo4j.exceptions.ClientError as e_alt: # If SHOW VECTOR INDEXES is not supported or fails
                    print(f"Alternative dimension fetch for '{index_name}' failed: {e_alt}")

                print(f"Warning: Vector index '{index_name}' not found or its dimension property is missing/unrecognized in db.indexes() or SHOW VECTOR INDEXES.")
                return None
        except neo4j.exceptions.ClientError as e:
            print(f"Error fetching dimension for vector index '{index_name}': {e}")
            return None


    async def _execute_write(self, query: str, params: dict = None):
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            return await result.data() # Consume and get data

    async def _execute_read(self, query: str, params: dict = None):
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            return await result.data() # Consume and get data

    async def add_session_to_memory(self, session: Session) -> None:
        """Ingests the given session's events into long-term memory storage."""
        await self._ensure_indexes()
        async with self._semaphore:
            # For each event in the session, mark it as memory if it has text content
            parameters = {"app": session.app_name, "user": session.user_id, "sid": session.id, "events": []}
            for evt in session.events:
                text_content = ""
                if evt.content:
                    try:
                        parts = evt.content.parts if hasattr(evt.content, "parts") else []
                        text_content = " ".join(p.text for p in parts if hasattr(p, "text")) if parts else str(evt.content)
                    except Exception:
                        text_content = str(evt.content)
                if not text_content: # Skip events with no text
                    continue
                
                embedding = None
                if self._embedding_fn and self._vector_dim: # Use configured _vector_dim for creating embedding
                    embedding_vec = self._embedding_fn(text_content)
                    if embedding_vec is not None:
                        embedding = [float(x) for x in embedding_vec]
                
                parameters["events"].append({
                    "eid": evt.id,
                    "text": text_content,
                    "author": evt.author,
                    "ts": evt.timestamp,
                    "embedding": embedding
                })
            if not parameters["events"]:
                return
            
            query = """
            UNWIND $events AS chunk_data
            MERGE (mc:MemoryChunk {eid: chunk_data.eid})
            SET mc.text = chunk_data.text,
                mc.author = chunk_data.author,
                mc.ts = chunk_data.ts,
                mc.app_name = $app,
                mc.user_id = $user,
                mc.session_id = $sid
            WITH mc, chunk_data
            WHERE chunk_data.embedding IS NOT NULL
            SET mc.embedding = chunk_data.embedding
            RETURN count(mc) AS added
            """
            await self._execute_write(query, parameters)

    async def search_memory(self, *, app_name: str, user_id: str, query: str, config: dict = None) -> SearchMemoryResponse:
        """Searches the memory for the given app/user for relevant information."""
        await self._ensure_indexes()
        
        vector_mode = bool(self._embedding_fn and self._vector_dim)
        records = []

        if vector_mode:
            # Ensure dimension is fetched and cached if not already
            # This uses the _index_creation_lock to ensure thread-safe cache population
            if self._mem_idx_dim_cache is None and self._vector_dim:
                async with self._index_creation_lock:
                    if self._mem_idx_dim_cache is None: # Double check lock
                        fetched_dim = await self._get_vector_index_dimension('MemoryVectorIndex')
                        if fetched_dim:
                            self._mem_idx_dim_cache = fetched_dim
                        else:
                            print(f"Warning: Could not determine dimension for 'MemoryVectorIndex' from DB. "
                                  f"Falling back to configured dimension: {self._vector_dim} for validation if needed. "
                                  "Vector search might be impaired or fail if index doesn't exist or dimension is wrong.")
                            # If index truly doesn't exist, vector query will fail.
                            # If it exists but dim fetch failed, using configured dim for validation is a guess.
                            self._mem_idx_dim_cache = int(self._vector_dim) if self._vector_dim else None


            query_embedding_list = None
            if self._embedding_fn:
                try:
                    query_vec_raw = self._embedding_fn(query)
                    if query_vec_raw is not None:
                        query_embedding_list = [float(x) for x in query_vec_raw]
                except Exception as e:
                    print(f"Error generating embedding for query: {e}")
                    query_embedding_list = None
            
            if query_embedding_list:
                if self._mem_idx_dim_cache is not None: # Actual dimension from DB or fallback
                    if len(query_embedding_list) != self._mem_idx_dim_cache:
                        raise ValueError(
                            f"Query embedding dimension ({len(query_embedding_list)}) "
                            f"does not match index dimension ({self._mem_idx_dim_cache}) for 'MemoryVectorIndex'."
                        )
                elif self._vector_dim : # Configured dim exists, but DB cache failed
                     if len(query_embedding_list) != int(self._vector_dim): # Validate against configured as last resort
                        raise ValueError(
                            f"Query embedding dimension ({len(query_embedding_list)}) "
                            f"does not match configured vector_dimension ({self._vector_dim}). "
                            "And could not verify actual index dimension from DB."
                        )
                # If _mem_idx_dim_cache is None and self._vector_dim is also None, vector search shouldn't be attempted.

                if self._mem_idx_dim_cache is not None or self._vector_dim is not None: # Proceed if we have some dimension to work with
                    vec_query_cypher = (
                        "CALL db.index.vector.queryNodes('MemoryVectorIndex', $k, $qVec) YIELD node AS mc, score "
                        "WHERE mc.app_name = $app_name AND mc.user_id = $user_id "
                        "RETURN mc.session_id AS session_id, mc.eid AS event_id, mc.text AS text, mc.author AS author, mc.ts AS ts, score"
                    )
                    params_vec = {"k": self._top_k, "qVec": query_embedding_list, "app_name": app_name, "user_id": user_id}
                    try:
                        vec_records_data = await self._execute_read(vec_query_cypher, params_vec)
                        if self._vector_threshold is not None:
                            vec_records_data = [r for r in vec_records_data if r["score"] >= self._vector_threshold]
                        records.extend(vec_records_data)
                    except neo4j.exceptions.ClientError as e:
                        print(f"Vector search query failed: {e}. This might be due to missing/misconfigured 'MemoryVectorIndex'.")
                        # Optionally, disable vector_mode for this call if query fails
                        # vector_mode = False
            else: # query_embedding_list is None
                print("Could not generate query embedding. Skipping vector search.")


        # Full-text search (fallback or primary if vector search not applicable/failed)
        text_query_cypher = (
            "CALL db.index.fulltext.queryNodes('MemoryTextIndex', $query) YIELD node AS mc, score "
            "WHERE mc.app_name = $app_name AND mc.user_id = $user_id "
            "RETURN mc.session_id AS session_id, mc.eid AS event_id, mc.text AS text, mc.author AS author, mc.ts AS ts, score"
        )
        try:
            text_records_data = await self._execute_read(text_query_cypher, {"query": query, "app_name": app_name, "user_id": user_id})
            records.extend(text_records_data)
        except neo4j.exceptions.ClientError as e:
            print(f"Full-text search query failed: {e}. This might be due to missing 'MemoryTextIndex'.")


        unique_records_map = {}
        for rec in records: # records now contains both vector and text results
            event_id_key = rec.get("event_id")
            if event_id_key not in unique_records_map:
                 unique_records_map[event_id_key] = rec
            else:
                if rec.get("score", 0) > unique_records_map[event_id_key].get("score", 0):
                    unique_records_map[event_id_key] = rec

        sessions_map = {}
        for rec_key, rec_data in unique_records_map.items():
            sid = rec_data.get("session_id", "unknown_session_id")
            if sid not in sessions_map:
                sessions_map[sid] = {"session_id": sid, "events_data": []}
            sessions_map[sid]["events_data"].append(rec_data)


        memory_results_list = []
        for sid, data in sessions_map.items():
            events_for_session = []
            for rec_data_item in data["events_data"]:
                event_content = Content(parts=[Part(text=rec_data_item.get("text", ""))])
                event_for_memory = Event(
                    id=rec_data_item.get("event_id", Event.new_id()),
                    author=rec_data_item.get("author", "unknown"),
                    timestamp=rec_data_item.get("ts", 0.0),
                    content=event_content
                )
                events_for_session.append(event_for_memory)
            
            if events_for_session:
                memory_results_list.append(MemoryResult(session_id=sid, events=events_for_session))
            
        return SearchMemoryResponse(memories=memory_results_list)

    def search_memory_sync(self, *, app_name: str, user_id: str, query: str, config: dict = None) -> SearchMemoryResponse:
        """Synchronous wrapper for search_memory."""
        return asyncio.run(self.search_memory(app_name=app_name, user_id=user_id, query=query, config=config))

    async def close(self):
        """Close the Neo4j driver connection."""
        if self._driver:
            await self._driver.close()