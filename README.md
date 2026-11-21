# F1 Regulations & History Q&A

A Retrieval-Augmented Generation system that answers natural-language questions about Formula 1 regulations and history, with every answer grounded in cited source material rather than model memory alone.

**Live Demo**: [https://f1-rag-api.onrender.com/docs](https://f1-rag-api.onrender.com/docs)

## What it does

Ask a question like *"What is the minimum length of the forward piece of the Plank in the 2027 Technical Regulations?"* and get back an answer with numbered, traceable citations — not just a confident-sounding guess.

## Corpus

57 source documents, spanning two very different document types:

- **16 FIA Formula 1 regulation PDFs** (2021-2027), including multiple issues/revisions of the same season's regulations
- **41 Wikipedia articles** covering seasons, teams, drivers, circuits, and technical concepts

## Architecture

- **Chunking** — header-aware splitting for regulation PDFs, recursive character splitting for Wikipedia text (`langchain-text-splitters`)
- **Embeddings** — Voyage AI
- **Vector store** — Qdrant Cloud
- **Retrieval + generation** — a custom LangChain `BaseRetriever` wrapping the Qdrant/Voyage calls, composed via LCEL with `ChatAnthropic` (Claude) and a `StrOutputParser`
- **Observability** — Langfuse tracing
- **API** — FastAPI, exposing `/ask` and `/health`
- **Deployment** — Docker container on Render

## Observability

The question-answering pipeline is traced with [Langfuse](https://langfuse.com) (Python SDK v3). Each call to `answer_question()` produces one trace:

- a parent span, `answer_question`, with the question and `top_k` as input and the answer plus cited sources as output
- the retriever call (query and retrieved chunks), attached through the LangChain `CallbackHandler`
- the prompt and Claude call, attached through the same handler

This makes it possible to tell whether a bad answer came from retrieval (wrong or missing chunks) or from generation (right chunks, wrong answer). The evaluation script (`eval.py`) calls the retriever and chain directly, so evaluation runs are not traced.

Set these in `.env` to enable it:

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Evaluation

### Golden set

`golden_set.json` holds 27 hand-built questions (including deliberately adversarial cases across near-duplicate regulation documents). Each entry has:

- `question`
- `expected_sources` — one or more source documents that correctly answer it
- `expected_answer` — a short answer written by hand from the source documents

### Running the evaluation

```
uv run python eval.py
```

For each question, the script retrieves the top-k chunks (`K` is set at the top of the file, default 5), generates an answer from those same chunks, and writes the question, expected answer, generated answer, retrieved contexts and rank of the first correct source to `results.json`. It prints Hit Rate and MRR and lists any questions whose expected source was not retrieved.

### Retrieval results

| k | Hit Rate | MRR |
| --- | -------- | ----- |
| 1 | 0.852 | 0.852 |
| 3 | 1.000 | ≈0.91 |
| 5 | 1.000 | ≈0.92 |
| 10 | 1.000 | ≈0.91 |

The production API defaults to `top_k=5`, where every question in the golden set retrieves its correct source.

**Run-to-run variation:** MRR at k≥3 is not perfectly reproducible. Across repeated runs at k=5 it ranged from 0.914 to 0.938, while Hit Rate stayed at 1.000 and k=1 stayed at 0.852. Differences below about 0.025 MRR should be treated as noise. Several questions have near-duplicate source documents, so near-tied search results can swap order between runs.

**Caveat:** a subset of golden-set questions have multiple valid expected sources (e.g. a driver's own page and their team's page both correctly answer "which team do they race for"), which makes those questions somewhat easier to hit — worth reading the 100% figure at k≥3 with that in mind rather than as unqualified precision.

### Generation model

Answers are generated with the model set in the `ANTHROPIC_MODEL` environment variable (default `claude-haiku-5-5`).

## Known limitations

9 of the 17 regulation PDFs use FIA's redline/revision-tracking convention, which causes some numeric values to appear duplicated in extracted text for recently-changed passages. Answers touching those specific passages should be treated as lower-confidence; the golden evaluation set deliberately avoids ground-truth questions anchored to redline-affected text.

The corpus also contains several issues of the same regulations. Four golden-set questions miss at k=1 (all are recovered by k=3). They concern topics covered by multiple near-duplicate documents, such as several issues of the same regulation, so a sibling document can rank first.

## API

**`POST /ask`**

```
{
  "question": "Where is Silverstone located?",
  "top_k": 5
}
```

Response:

```
{
  "answer": "...",
  "sources": [
    {"n": 1, "url": "https://en.wikipedia.org/wiki/Silverstone_Circuit", "doc_type": "wikipedia"}
  ]
}
```

**`GET /health`** — returns `{"status": "ok"}` when the Qdrant collection is reachable.

## Running locally

```
uv sync --extra api
```

Create a `.env` with:

```
QDRANT_URL=...
QDRANT_API_KEY=...
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-haiku-5-5
VOYAGEAI_API_KEY=...
```

```
uv run uvicorn app.main:app --reload --port 8000
```

## Running with Docker

```
docker build -t f1-rag-api .
docker run -p 8080:8080 --env-file .env f1-rag-api
```

## Tech stack

Python · FastAPI · LangChain · Anthropic Claude · Voyage AI · Qdrant · Docker · uv
