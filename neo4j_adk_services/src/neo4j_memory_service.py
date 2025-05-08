import neo4j # Added for exception handling
from neo4j import GraphDatabase
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
                 vector_distance_threshold: float = None):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._embedding_fn = embedding_function
        self._vector_dim = vector_dimension
        self._top_k = similarity_top_k or 5
        self._vector_threshold = vector_distance_threshold
        # Create full-text index for memory content (if not exists)
        with self._driver.session(database=self._database) as db_session: # Renamed session to db_session
            try:
                db_session.run(
                    "CALL db.index.fulltext.createNodeIndex('MemoryTextIndex', ['MemoryChunk'], ['text'])"
                )
            except neo4j.exceptions.ClientError as e:
                if "An equivalent index already exists" not in str(e) and "already exists" not in str(e):
                    raise
            
            # If vector search is desired, create vector index
            if self._vector_dim:
                try:
                    db_session.run(
                        "CALL db.index.vector.createNodeIndex('MemoryVectorIndex', 'MemoryChunk', 'embedding', $dim, 'cosine')",
                        {"dim": int(self._vector_dim)}
                    )
                except neo4j.exceptions.ClientError as e:
                    if "An equivalent index already exists" not in str(e) and "already exists" not in str(e):
                        raise
        super().__init__()

    def _execute_write(self, query: str, params: dict = None):
        with self._driver.session(database=self._database) as session:
            return session.run(query, params).data()

    def _execute_read(self, query: str, params: dict = None):
        with self._driver.session(database=self._database) as session:
            return session.run(query, params).data()

    def add_session_to_memory(self, session: Session) -> None:
        """Ingests the given session's events into long-term memory storage."""
        # For each event in the session, mark it as memory if it has text content
        query_parts = []
        parameters = {"app": session.app_name, "user": session.user_id, "sid": session.id, "events": []}
        # We will batch update events; collect IDs and optionally embeddings
        for evt in session.events:
            # Derive the same text field as in SessionService append_event
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
            if self._embedding_fn and self._vector_dim:
                embedding_vec = self._embedding_fn(text_content)
                if embedding_vec is not None:
                    embedding = [float(x) for x in embedding_vec]
            
            # Add event data to parameters for UNWIND
            # The patch uses 'bulk' as the parameter name for UNWIND, and 'row' for each item.
            # The original code used 'events' and 'memEvt'. Sticking to 'events' for now.
            # The patch also adds 'eid' to the chunk properties.
            parameters["events"].append({
                "eid": evt.id, # Event ID for linking or identification
                "text": text_content,
                "author": evt.author, # Added author as per patch example
                "ts": evt.timestamp,   # Added timestamp as per patch example
                "embedding": embedding
            })
        if not parameters["events"]:
            return
        # Build a query to set Memory label (and embedding) on each event
        # We'll use UNWIND for batch operation
        # Corrected author matching logic
        author_match_clause = "e.author = $author_val"
        if hasattr(session, 'agent') and hasattr(session.agent, 'name'):
            parameters["author_val"] = session.agent.name
        else:
            parameters["author_val"] = session.user_id

        # The patch uses 'MemoryChunk' label and different properties.
        # Adapting to use 'MemoryChunk' and the properties from the patch.
        # The original code set :Memory label on existing :Event nodes.
        # The patch implies creating new :MemoryChunk nodes.
        # For consistency with the patch's schema section, let's use MemoryChunk.
        # This means we are not just labeling existing events but creating distinct memory representations.
        
        # If MemoryChunks are distinct from Events, we need a way to link them back or store enough context.
        # The patch's MemoryChunk properties: text, author, ts, eid (event id).
        # The UNWIND $bulk AS row MERGE (mc:MemoryChunk {text: row.text, author: row.author, ts: row.ts, eid: row.eid})
        # This implies MemoryChunk nodes are identified by these properties if MERGE is to work as expected for deduplication.
        # Or, if eid is unique for MemoryChunk, MERGE (mc:MemoryChunk {eid: row.eid}) SET mc += row
        
        # Let's assume MemoryChunk nodes are created, and eid links back to the original event.
        # The patch's schema has `CALL db.index.fulltext.createNodeIndex('memoryTextIdx', ['MemoryChunk'], ['text']);`
        # and `CREATE VECTOR INDEX memoryVectorIdx FOR (m:MemoryChunk) ON (m.embedding)`
        
        query = """
        UNWIND $events AS chunk_data // Changed from $bulk to match parameter name
        // Create a new MemoryChunk node for each event to be memorized.
        // If eid is meant to be unique for MemoryChunk:
        MERGE (mc:MemoryChunk {eid: chunk_data.eid})
        SET mc.text = chunk_data.text,
            mc.author = chunk_data.author,
            mc.ts = chunk_data.ts,
            mc.app_name = $app,     // Store app_name for partitioning
            mc.user_id = $user,     // Store user_id for partitioning
            mc.session_id = $sid    // Store session_id
        // Optionally set embedding if present
        WITH mc, chunk_data
        WHERE chunk_data.embedding IS NOT NULL
        SET mc.embedding = chunk_data.embedding
        RETURN count(mc) AS added
        // (optional) pre-compute similarity edges offline:
        // MATCH (mc1:MemoryChunk), (mc2:MemoryChunk)
        // WHERE id(mc1) < id(mc2) AND mc1.app_name = mc2.app_name AND mc1.user_id = mc2.user_id // Partitioned similarity
        // WITH mc1, mc2, gds.similarity.cosine(mc1.embedding, mc2.embedding) AS score
        // WHERE score > 0.8 // Similarity threshold
        // CREATE (mc1)-[:SIMILAR {score:score}]->(mc2)
        """
        # No session_id needed in params if MemoryChunks are not directly linked to Session nodes in this query.
        # App and User are passed for partitioning MemoryChunks.
        self._execute_write(query, parameters)
        # MemoryService usually doesn't return anything; it's a fire-and-forget ingestion.

    def search_memory(self, *, app_name: str, user_id: str, query: str, config: dict = None) -> SearchMemoryResponse:
        """Searches the memory for the given app/user for relevant information."""
        results = []
        # import json # Not strictly needed here anymore unless for other parts
        # Decide on search strategy: vector or fulltext
        vector_mode = bool(self._embedding_fn and self._vector_dim)
        records = []
        if vector_mode:
            # Perform vector search
            query_vector = None
            try:
                query_vec = self._embedding_fn(query)
                query_vector = [float(x) for x in query_vec] if query_vec is not None else None
            except Exception as e:
                # In a real app, log this error: print(f"Error generating embedding: {e}")
                query_vector = None
            if query_vector:
                vec_query = (
                    "CALL db.index.vector.queryNodes('MemoryVectorIndex', $k, $qVec) YIELD node AS mc, score "
                    "WHERE mc.app_name = $app_name AND mc.user_id = $user_id "
                    "RETURN mc.session_id AS session_id, mc.eid AS event_id, mc.text AS text, mc.author AS author, mc.ts AS ts, score"
                )
                params = {"k": self._top_k, "qVec": query_vector, "app_name": app_name, "user_id": user_id}
                vec_records = self._execute_read(vec_query, params)
                if self._vector_threshold is not None:
                    vec_records = [r for r in vec_records if r["score"] >= self._vector_threshold]
                records.extend(vec_records)

        text_query = (
            "CALL db.index.fulltext.queryNodes('MemoryTextIndex', $query) YIELD node AS mc, score "
            "WHERE mc.app_name = $app_name AND mc.user_id = $user_id "
            "RETURN mc.session_id AS session_id, mc.eid AS event_id, mc.text AS text, mc.author AS author, mc.ts AS ts, score"
        )
        text_records = self._execute_read(text_query, {"query": query, "app_name": app_name, "user_id": user_id})
        
        unique_records_map = {}
        # Combine records, prioritizing vector search results if IDs match, or higher score
        # For simplicity, let's ensure event_id is the primary key for uniqueness of a memory item
        for rec in records + text_records:
            event_id_key = rec.get("event_id")
            if event_id_key not in unique_records_map:
                 unique_records_map[event_id_key] = rec
            else:
                # If duplicate (same event_id), prefer the one with higher score
                # This assumes scores from vector and text are somewhat comparable or one is primary
                if rec.get("score", 0) > unique_records_map[event_id_key].get("score", 0):
                    unique_records_map[event_id_key] = rec

        sessions_map = {}
        for rec_key, rec_data in unique_records_map.items(): # Iterate over unique records
            sid = rec_data.get("session_id")
            if not sid:
                # This case should ideally not happen if session_id is always stored and returned
                # Fallback or error handling might be needed here
                sid = "unknown_session_id" # Placeholder

            evt_snippet = f"{rec_data.get('author', 'unknown_author')}: {rec_data.get('text', '')}"
            if sid not in sessions_map:
                sessions_map[sid] = {"session_id": sid, "snippets": []}
            
            # Avoid duplicate snippets within the same session_id grouping
            if evt_snippet not in sessions_map[sid]["snippets"]:
                 sessions_map[sid]["snippets"].append(evt_snippet)

        memory_results_list = []
        for sid, data in sessions_map.items():
            # data["snippets"] now contains raw text snippets like "author: text"
            # We need to convert these back into Event objects for MemoryResult
            # Each 'rec_data' that contributed to 'data' had 'event_id', 'author', 'ts', 'text'
            
            # Reconstruct events for this session_id from the unique_records_map
            # This is a bit indirect. The 'data' in sessions_map only has snippets.
            # We need to go back to unique_records_map to get full event details for this session_id.
            
            events_for_session = []
            for rec_key, rec_data_item in unique_records_map.items():
                if rec_data_item.get("session_id") == sid:
                    # Create a minimal Event object
                    event_content = Content(parts=[Part(text=rec_data_item.get("text", ""))])
                    event_for_memory = Event(
                        id=rec_data_item.get("event_id", Event.new_id()), # Use original eid or new if missing
                        author=rec_data_item.get("author", "unknown"),
                        timestamp=rec_data_item.get("ts", 0.0),
                        content=event_content
                        # Actions are not typically stored in MemoryChunk, so not reconstructed here
                    )
                    events_for_session.append(event_for_memory)
            
            if events_for_session: # Only add if there are events
                memory_results_list.append(MemoryResult(session_id=sid, events=events_for_session))
            
        return SearchMemoryResponse(memories=memory_results_list)

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()