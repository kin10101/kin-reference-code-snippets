# Neo4j GraphRAG Pipeline — Knowledge Transfer

---

## 1. What Is GraphRAG?

**Standard RAG (Retrieval-Augmented Generation)** retrieves text chunks similar to a query via vector search and feeds them to an LLM.

**GraphRAG** goes further: it also builds a **knowledge graph** of entities (chips, pins, peripherals, APIs, …) and their relationships. At query time, the graph provides *structured relational context* that pure vector search cannot capture.

```
Standard RAG:   PDF → Chunks → Embeddings → Vector DB → LLM answer
GraphRAG:       PDF → Chunks → Embeddings → Vector DB    ┐
                           └── Entity Extraction → Graph DB ┘ → LLM answer
```

<div style="width:500px">

![image.png](/.attachments/image-d675beb3-d0b8-4930-b797-8cbca55c62cc.png)

</div>

This pipeline combines **both** into a **hybrid** approach, giving the LLM three types of context simultaneously:

| Source | What it provides |
|--------|-----------------|
| **Vector search** | Semantically similar text chunks |
| **Full-text / keyword search** | Exact keyword matches (Lucene) |
| **Graph traversal** | Entity relationships (e.g., which pins belong to a chip) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / app entry                          │
│   cli.py  ──────────►  ingest_pdfs()  /  query_knowledge_graph()│
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   LangGraph StateGraph  │
              │  (langgraph_flow/)      │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼──────────────────────┐
         │                 │                      │
   INGESTION PATH    QUERY PATH            (shared state)
         │                 │
   ┌─────▼──────┐    ┌─────▼──────┐
   │ node_setup │    │node_answer │
   │ (indexes)  │    │_question   │
   └─────┬──────┘    └─────┬──────┘
   ┌─────▼──────┐          │
   │node_load   │    ┌─────▼──────────────────┐
   │_pdfs       │    │  retrieval/             │
   └─────┬──────┘    │  hybrid_retriever.py    │
   ┌─────▼──────┐    │  + qa_chain.py          │
   │node_chunk  │    └─────────────────────────┘
   └─────┬──────┘
   ┌─────▼──────┐
   │node_build  │
   │_graph      │
   └─────┬──────┘
         │
   ┌─────▼──────────────────────────────────┐
   │  graph/                                │
   │  pdf_loader → chunker → graph_builder  │
   └─────┬──────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────┐
   │           Neo4j Aura / Local           │
   │   Lexical graph  +  Entity graph       │
   │   + Vector index  +  Full-text index   │
   └────────────────────────────────────────┘
```

<div style="width:400px">

![image.png](/.attachments/image-993b7e22-981b-4d83-8838-9dddfc065408.png)
</div> 

---

## 3. Repository Layout

```
neo4j_graphrag_pipeline/
│
├── cli.py                      # Rich CLI (ingest / query / chat / cache-status)
├── main.py                     # Alternative script entry point
├── pyproject.toml              # Package metadata & dependencies
│
├── core/
│   ├── config.py               # Central Config singleton (env vars)
│   ├── logger.py               # Two loggers: pipeline_log, graph_log
│   └── neo4j_client.py         # Shared Neo4j driver & LangChain graph singletons
│
├── graph/
│   ├── pdf_loader.py           # PDF → LangChain Document objects
│   ├── chunker.py              # Document → overlapping text chunks
│   └── graph_builder.py        # Chunks → Neo4j lexical + entity graph + vectors
│
├── retrieval/
│   ├── hybrid_retriever.py     # Vector + full-text + graph entity search
│   └── qa_chain.py             # LCEL chain: retrieve → prompt → LLM → answer
│
└── langgraph_flow/
    ├── state.py                # PipelineState TypedDict
    ├── nodes.py                # Node functions + conditional routers
    └── graph.py                # Compiled StateGraph singletons
```

---

## 4. Component Deep Dives

### 4.1 Config & Core Utilities

**`core/config.py`** — A single `Config` dataclass (or Pydantic settings model) that reads all environment variables in one place:

| Variable | Purpose |
|----------|---------|
| `NEO4J_URI` | Bolt/HTTPS URI for Neo4j |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Auth credentials |
| `NEO4J_DATABASE` | Target database name |
| `OPENAI_API_KEY` | OpenAI key for LLM extraction |
| `LLM_MODEL` | Model name (e.g., `gpt-4o`) |
| `EMBEDDING_MODEL` | HuggingFace model name for embeddings |
| `VECTOR_INDEX_NAME` | Name of the Neo4j vector index |

**`core/neo4j_client.py`** — Lazy-initialized singletons so the driver is created once and reused:

```python
get_driver()          # Returns a raw neo4j.Driver
get_langchain_graph() # Returns a LangChain Neo4jGraph wrapper
ensure_indexes()      # Creates vector + full-text indexes if they don't exist
reset_clients()       # Closes and clears cached connections (used on retry)
```

The client also exposes `is_retryable_neo4j_error()` to detect transient failures (DNS errors, routing failures, connection resets) that are worth retrying.

**`core/logger.py`** — Two separate loggers used *consistently* throughout the codebase:

| Logger | Used in | Purpose |
|--------|---------|---------|
| `pipeline_log` | `langgraph_flow/nodes.py` only | High-level step progress |
| `graph_log` | `graph/` and `retrieval/` modules | Technical detail logging |

> **Rule:** Node functions only call `pipeline_log`. Low-level modules only call `graph_log`. This separation keeps log output clean and structured.

---

### 4.2 PDF Loader

**`graph/pdf_loader.py`**

Accepts a mixed list of file paths and directory paths. Directories are scanned recursively for `*.pdf` files.

```
Input:  ["docs/", "manual.pdf"]
           │
           ▼
  _iter_resolved_pdf_files()   ← de-duplicates, skips non-PDFs
           │
           ▼
  PyPDFLoader (per file)       ← one Document per page
           │
           ▼
  _infer_mcu_family(path)      ← tags each doc with "TC3xx" / "TC4xx" / "unknown"
           │
           ▼
Output: List[Document]  (metadata: source, page, mcu_family, doc_id)
```

<div style="width:400px">

![image.png](/.attachments/image-db590e7d-bcd3-4cff-ae80-89602fd86894.png)
</div> 

The `mcu_family` tag is added to every `Document`'s metadata so that later retrieval can filter by MCU family when the user's question mentions a specific chip family.

---

### 4.3 Chunker

**`graph/chunker.py`**

Three-layer filtering system before splitting:

```
Layer 1 — Page-level filter
  Is the first 300 chars a boilerplate header (TOC, disclaimer, revision history)?
  → Drop entire page

Layer 2 — Text cleaning
  Strip TOC dot-leaders ("Chapter 1 ....... 5")
  Strip repeated separators (----, ....., ____)
  Collapse excessive blank lines

Layer 3 — Short-chunk filter (post-split)
  Discard any chunk shorter than MIN_CHUNK_LENGTH (100 chars)
```

<div style="width:250px">

![image.png](/.attachments/image-905849dc-893d-4da6-b0b8-37d696c02501.png)
</div> 

The splitter is `RecursiveCharacterTextSplitter` configured with:
- **Chunk size** measured in **tokens** (cl100k_base via tiktoken), not characters
- **Chunk overlap** to preserve context across chunk boundaries
- Both values come from `cfg.CHUNK_SIZE` / `cfg.CHUNK_OVERLAP`

Each chunk gets a **deterministic SHA-256 hash ID** based on its text content, enabling idempotent re-ingestion.

---

### 4.4 Graph Builder

**`graph/graph_builder.py`** — The most complex module. Executes four sub-steps:

#### Step A — Lexical Graph

Creates the `Document → Chunk` structure in Neo4j:

```
(:Document {id, source, mcu_family})
    -[:HAS_CHUNK]->
(:Chunk {id, text, page, source, mcu_family})
```

This is written directly via Cypher (not via LangChain helpers) for fine-grained control.

#### Step B — Entity Extraction (LLM)

For each chunk, the LLM is asked to extract **entities** and **relationships** using a detailed system prompt and **Pydantic structured output**:

```python
class ExtractedEntity(BaseModel):
    id: str          # Exact name as written in text
    label: str       # PascalCase (e.g., Chip, Pin, Peripheral)
    properties: dict[str, str]  # Only explicitly stated values

class ExtractedRelationship(BaseModel):
    source_id: str
    target_id: str
    type: str        # SCREAMING_SNAKE_CASE (e.g., HAS_PIN, MANUFACTURED_BY)
    properties: dict[str, str]

class GraphExtractionOutput(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
```

The LLM is called via `llm.with_structured_output(GraphExtractionOutput, method="function_calling")`.

> **Why function_calling instead of strict JSON mode?**  
> OpenAI's strict JSON-schema mode does not support `dict[str, str]` (free-form additional properties). Function-calling mode has no such restriction.

#### Step C — Entity Graph

The Pydantic output is converted to LangChain `GraphDocument` objects and written to Neo4j via `Neo4jGraph.add_graph_documents()`. Neo4j `MERGE` is used so the same entity extracted from multiple chunks is deduplicated into a single node.

```
(:Chip {id: "TC387"})
    -[:HAS_PERIPHERAL]->
(:Peripheral {id: "GTM"})
    -[:HAS_PORT]->
(:Peripheral {id: "TOM"})
```

#### Step D — Vector Embeddings

Chunk text is embedded with the configured HuggingFace model and the vectors are stored in a `Neo4jVector` index on the `Chunk` nodes:

```
(:Chunk {text: "...", embedding: [0.12, -0.34, ...]})
```

<div style="width:500px">

![image.png](/.attachments/image-c3a0a207-14dd-4842-aa90-9d78bb1564e9.png)
</div> 


#### Caching & Resilience

The graph builder caches extraction results to disk (`run_artifacts/graphrag_cache/`) as gzip-compressed JSONL files. If ingestion is interrupted, re-running it will skip already-processed chunks (checkpoint resume). Transient Neo4j errors trigger automatic retry with exponential back-off.

<div style="width:350px">

![image.png](/.attachments/image-9764f69c-7331-42ce-98ce-62c060568241.png)
</div> 

---

### 4.5 Hybrid Retriever

**`retrieval/hybrid_retriever.py`**

At query time, three strategies run **independently** and their results are merged into a single context string:

```
User question
      │
      ├──► 1. Vector Search
      │         Embed the question → cosine similarity against Chunk.embedding
      │         Returns top-K most semantically similar chunks
      │
      ├──► 2. Full-Text (Lucene) Search
      │         Sanitize query → call Neo4j fulltext index on Chunk.text
      │         Returns keyword-matching chunks
      │
      └──► 3. Graph Entity Search
                Extract entity names from question → MATCH in Neo4j
                Return nodes + their 1-hop relationships as structured facts
                
            Merge all results
                  │
                  ▼
          Formatted context string:
            [Filter] mcu_family=TC4xx
            [Chunk 1] ...
            [Chunk 2] ...
            [Keyword 1] ...
            [Graph] TC387 -[:HAS_PERIPHERAL]-> GTM ...
```
<div style="width:500px">

![image.png](/.attachments/image-0390bc28-1d96-4d4b-8fd7-2599dfef6cae.png)
</div> 



**MCU Family Detection:**  
The retriever calls `detect_mcu_family(query)` which uses regex to detect `TC3xx` or `TC4xx` patterns in the question. When detected, vector and graph searches are filtered to that family only.

**Lucene Query Sanitization:**  
Special characters (`?`, `/`, `\`, operators like `+`, `-`, `!`) are escaped or stripped before being passed to Neo4j's full-text index to prevent query parse errors.

---

### 4.6 QA Chain

**`retrieval/qa_chain.py`**

A simple **LCEL (LangChain Expression Language)** chain:

```python
chain = (
    {
        "context": lambda x: hybrid_retrieve(x["question"]),
        "question": lambda x: x["question"],
    }
    | PROMPT       # ChatPromptTemplate with system instructions
    | llm          # Azure OpenAI / OpenAI / Anthropic (env-configurable)
    | StrOutputParser()
)
```

The system prompt instructs the LLM to:
1. Answer **only** from the provided context
2. Treat `[Chunk]` / `[Keyword]` lines as primary factual sources
3. Use `[Graph]` lines for relationship understanding
4. Respect any `[Filter]` scope restriction
5. Say *"I don't know"* if the answer is not in the context

**LLM Provider Selection** (via `LLM_PROVIDER` env var):

| Value | LLM Used |
|-------|---------|
| `openai` (default) | `ChatOpenAI` (gpt-4o) |
| `azure` | `AzureChatOpenAI` |
| `anthropic` | `ChatAnthropic` |

---

## 5. LangGraph State Machine

The pipeline uses **LangGraph** to orchestrate multi-step workflows as explicit state machines, making the execution flow visible, testable, and resumable.

### 5.1 Ingestion Graph

```
           ┌─────────┐
START ────► │  setup  │  (ensure Neo4j indexes exist)
           └────┬────┘
                │ error? → END
           ┌────▼─────────┐
           │  load_pdfs   │  (PDF → Document objects)
           └────┬─────────┘
                │ error? → END
           ┌────▼────┐
           │  chunk  │  (Document → text chunks)
           └────┬────┘
                │ error? → END
           ┌────▼────────────┐
           │  build_graph    │  (lexical + entity + vector)
           └────┬────────────┘
                │
               END
```
<div style="width:400px">

![image.png](/.attachments/image-477b45e5-b41e-4392-9086-62a832680491.png)
</div> 


Each edge is **conditional**: if `state["error"]` is set by any node, the graph short-circuits to `END` and the error is surfaced to the caller. This means a failure in step 2 does not attempt steps 3 and 4.

### 5.2 Query Graph

```
START ────► node_answer_question ────► END
```

Intentionally simple — a single node that calls the hybrid retriever and QA chain. The surrounding orchestration overhead is kept minimal for low-latency queries.

<div style="width:300px">

![image.png](/.attachments/image-98b3f583-ce53-46b4-9a10-0408f61d4a39.png)
</div> 

### 5.3 Shared State (`PipelineState`)

All nodes read from and write to a single `TypedDict`:

```python
class PipelineState(TypedDict, total=False):
    # Inputs
    pdf_paths: List[str]

    # Ingestion metrics (populated progressively)
    num_pages: Optional[int]
    num_chunks: Optional[int]
    num_graph_docs: Optional[int]
    ingestion_complete: bool

    # Query
    question: Optional[str]
    answer: Optional[str]

    # Control
    error: Optional[str]
    status: Optional[str]

    # Intermediate (not returned to callers)
    _docs: Optional[Any]
    _chunks: Optional[Any]
```

Nodes are **pure functions** — they never mutate state in place; they always return a new dict using `{**state, "key": new_value}`.

---

## 6. Neo4j Data Model

The knowledge graph has two layers:

### Lexical Layer (structural)

```
(:Document {id, source, mcu_family, num_pages})
    │
    └──[:HAS_CHUNK]──►(:Chunk {
            id,           ← SHA-256 hash of text
            text,         ← raw chunk content
            page,         ← page number
            source,       ← original PDF path
            mcu_family,   ← "TC3xx" | "TC4xx" | "unknown"
            embedding     ← float[] vector (HuggingFace)
        })
```
<div style="width:200px">

![image.png](/.attachments/image-382f003d-6978-4c2f-8e40-9312eecab255.png)
</div> 

### Entity Layer (semantic)

```
(:Chip {id: "TC387", manufacturer: "Infineon", core: "TriCore"})
    │
    ├──[:HAS_PERIPHERAL]──►(:Peripheral {id: "GTM"})
    │                             │
    │                             └──[:HAS_PORT]──►(:Peripheral {id: "TOM"})
    │
    └──[:PACKAGED_AS]──────►(:Package {id: "BGA-292"})

(:EvaluationKit {id: "KIT_A2G_TC387_5V_TFT"})
    │
    └──[:DESIGNED_FOR]─────►(:Chip {id: "TC387"})
```

<div style="width:550px">

![image.png](/.attachments/image-4716147d-35af-4331-94a8-30364f7b0ee7.png)
</div> 

### Indexes

| Name | Type | On |
|------|------|----|
| `chunk_vector` | Vector (cosine) | `Chunk.embedding` |
| `chunk_fulltext` | Full-text (Lucene) | `Chunk.text` |

---

## 7. LLM Entity Extraction

The extraction system prompt (in `graph_builder.py`) is structured in 5 sections:

| Section | Content |
|---------|---------|
| **1 — What to Ignore** | Boilerplate to skip (TOC, legal, revision history…) |
| **2 — Domain Context** | ~30 pre-defined node label types across 4 domains |
| **3 — Entity Rules** | PascalCase IDs, consistency across chunks, no hallucination |
| **4 — Relationship Rules** | SCREAMING_SNAKE_CASE types, directional, no vague relationships |
| **5 — Final Instructions** | Precision over recall, empty output if boilerplate, stay dynamic |

The "stay dynamic" instruction is important: the model is told to use the domain context *as a guide*, not a hard limit. If the text describes something not in the list, it should invent the most descriptive PascalCase label.

**Anti-hallucination guardrails:**
- Properties: only extract values explicitly stated in the text
- Relationships: prefer 0 over a vague one
- Empty output: if a chunk is ambiguous or boilerplate, return empty lists

<img src="LLM Entity Extraction Pipeline.png" width="30%" />

---

## 8. CLI Reference

The CLI (`cli.py`) uses `argparse` with Rich-formatted output.

```bash
# Ingest one or more PDFs
uv run cli.py ingest path/to/doc.pdf
uv run cli.py ingest path/to/docs_folder/
uv run cli.py ingest docs/ extra/manual.pdf --no-langsmith

# Ask a single question
uv run cli.py query "What PWM peripherals does the TC387 have?"

# Interactive chat loop
uv run cli.py chat

# Inspect the ingestion cache
uv run cli.py cache-status
uv run cli.py cache-status --limit 10
```

**`--no-langsmith`** flag: automatically disables LangSmith tracing during ingestion (useful on restricted networks to prevent noise from failed telemetry calls).

**Cache Status** shows checkpoint and final-cache files in `run_artifacts/graphrag_cache/`, including file size and last-modified timestamp. Useful for debugging interrupted ingestion runs.

---

## 9. End-to-End Data Flow

### Ingestion (PDF → Graph)

```
User runs:  cli.py ingest TC387_datasheet.pdf
                │
                ▼
        ingest_pdfs(["TC387_datasheet.pdf"])
                │
                ▼ LangGraph ingestion_graph
    ┌───────────────────────────────────┐
    │ node_setup                        │
    │  └── ensure_indexes()             │
    │       Creates vector + fulltext   │
    │       indexes if missing          │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │ node_load_pdfs                    │
    │  └── load_pdfs([...])             │
    │       PyPDFLoader per file        │
    │       Tags mcu_family             │
    │       → 847 Document pages        │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │ node_chunk                        │
    │  └── chunk_documents([...])       │
    │       Filter boilerplate pages    │
    │       Clean text                  │
    │       Split (1000 tok / 200 ovlp) │
    │       Drop short chunks           │
    │       → 2,341 Chunk documents     │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │ node_build_graph                  │
    │  └── run_ingestion(chunks)        │
    │       A) Write lexical graph      │
    │          Document → Chunks        │
    │       B) LLM entity extraction   │
    │          (cached to disk)         │
    │       C) Write entity graph       │
    │          MERGE nodes/rels         │
    │       D) Embed + index chunks     │
    │       → 312 graph documents       │
    └───────────────────────────────────┘
                   │
                   ▼
             Neo4j database
        (lexical + entity + vectors)
```
<div style="width:530px">

![image.png](/.attachments/image-dd566efc-8b72-45e1-86d6-cadea868e36a.png)
</div> 


### Query (Question → Answer)

```
User asks:  "What TOM channels does the TC387 have?"
                │
                ▼
    query_knowledge_graph(question)
                │
                ▼ LangGraph query_graph
    ┌───────────────────────────────────┐
    │ node_answer_question              │
    │  └── answer_question(question)    │
    │       1. detect_mcu_family()      │
    │          → "TC3xx"                │
    │                                   │
    │       2. hybrid_retrieve()        │
    │          ┌── vector_search(q,5)   │
    │          ├── fulltext_search(q,5) │
    │          └── graph_search(q)      │
    │                                   │
    │       3. build_qa_chain()         │
    │          context + question       │
    │          → GPT-4o                 │
    │          → "TOM has 16 channels…" │
    └───────────────────────────────────┘
                │
                ▼
        Answer returned to user
```

<div style="width:530px">

![image.png](/.attachments/image-778137b8-efae-4fd0-898e-93b854dc9db4.png)
</div> 

---

## 10. Key Design Decisions

### Why LangGraph instead of a simple function call chain?

| Concern | LangGraph Solution |
|---------|-------------------|
| Error handling | Conditional edges short-circuit on error |
| Observability | Each node's state is logged and traceable |
| Testability | Each node is a pure function, easily unit-tested |
| Future parallelism | Nodes can be parallelized by adding edges |

### Why Pydantic structured output instead of `LLMGraphTransformer`?

`LLMGraphTransformer` (LangChain's built-in) uses a less-controlled extraction approach. Using direct Pydantic structured output with `method="function_calling"` gives:
- Full control over the system prompt
- Well-defined output schema validated at parse time
- Support for free-form `dict[str, str]` properties (blocked by strict JSON mode)

### Why HuggingFace embeddings instead of OpenAI?

- No per-token API cost for embeddings
- Can be self-hosted / run on-premise for sensitive documents
- Configurable via `EMBEDDING_MODEL` without code changes

### Why a three-layer irrelevant-section filter?

Technical PDFs are full of boilerplate (TOCs, legal notices, revision history). Passing these to the LLM for entity extraction:
1. Wastes API tokens on meaningless content
2. Pollutes the graph with low-quality nodes
3. Introduces noise into vector search results

The three-layer filter (page-level, text-cleaning, short-chunk) removes this before any LLM call.

### Caching & idempotent ingestion

Entity extraction is the most expensive step (one LLM call per chunk). The cache:
- Writes results to disk as gzip JSONL after each batch
- Uses SHA-256 chunk IDs as cache keys
- Allows interrupted ingestion to resume without re-processing completed chunks
