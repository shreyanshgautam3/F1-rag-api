# F1 Regulations & History Q&A

A Retrieval-Augmented Generation system that answers natural-language questions about Formula 1 regulations and history, with every answer grounded in cited source material rather than model memory alone.

**Live Demo**: [https://f1-rag-api.onrender.com/docs]

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
- **API** — FastAPI, exposing `/ask` and `/health`
- **Deployment** — Docker container on Render

## Retrieval evaluation

Retrieval quality is measured against a hand-built golden set of 27 questions (including deliberately adversarial cases across near-duplicate regulation documents), using Hit Rate and Mean Reciprocal Rank:

| k | Hit Rate | MRR |
|---|----------|-----|
| 1 | 0.815 | 0.815 |
| 3 | 0.963 | 0.889 |
| 5 | 1.000 | 0.898 |
| 7 | 1.000 | 0.898 |

The production API defaults to `top_k=5`, where every question in the golden set successfully retrieves its correct source.

**Caveat:** a subset of golden-set questions have multiple valid expected sources (e.g. a driver's own page and their team's page both correctly answer "which team do they race for"), which makes those questions somewhat easier to hit — worth reading the 100% figure at k≥5 with that in mind rather than as unqualified precision.

## Known limitations

9 of the 17 regulation PDFs use FIA's redline/revision-tracking convention, which causes some numeric values to appear duplicated in extracted text for recently-changed passages. Answers touching those specific passages should be treated as lower-confidence; the golden evaluation set deliberately avoids ground-truth questions anchored to redline-affected text.

## API

**`POST /ask`**
```json
{
  "question": "Where is Silverstone located?",
  "top_k": 5
}
```
Response:
```json
{
  "answer": "...",
  "sources": [
    {"n": 1, "url": "https://en.wikipedia.org/wiki/Silverstone_Circuit", "doc_type": "wikipedia"}
  ]
}
```

**`GET /health`** — returns `{"status": "ok"}` when the Qdrant collection is reachable.

## Running locally

```bash
uv sync --extra api
```

Create a `.env` with:
```
QDRANT_URL=...
QDRANT_API_KEY=...
ANTHROPIC_API_KEY=...
VOYAGEAI_API_KEY=...
```

```bash
uv run uvicorn app.main:app --reload --port 8000
```

## Running with Docker

```bash
docker build -t f1-rag-api .
docker run -p 8080:8080 --env-file .env f1-rag-api
```

## Tech stack

Python · FastAPI · LangChain · Anthropic Claude · Voyage AI · Qdrant · Docker · uv
