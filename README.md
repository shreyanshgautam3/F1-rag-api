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

This makes it possible to tell whether a bad answer came from retrieval (wrong or missing chunks) or from generation (right chunks, wrong answer). The evaluation scripts (`eval.py`, `judge.py`) call the retriever and chain directly, so evaluation runs are not traced.

Set these in `.env` to enable it:

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Evaluation

### Golden set

`golden_set.json` holds 27 hand-built questions spanning the regulation PDFs and the Wikipedia articles. Each entry has:

- `question`
- `expected_sources` — one or more source documents that correctly answer it
- `expected_answer` — a short answer written by hand from the source documents

### Running the evaluation

```
uv pip install -r requirements-eval.txt     # DeepEval, evaluation only (not needed by the API)
uv run python eval.py 5                     # retrieval + answers, writes results_k5.json
uv run python judge.py results_k5.json      # DeepEval scores, writes scores_k5.json
```

`eval.py` takes the top-k as an argument (default 5). For each question it retrieves the top-k chunks, generates an answer from those same chunks, and saves the question, expected answer, generated answer, retrieved contexts and rank of the first correct source. It prints Hit Rate and MRR and lists any questions whose expected source was not retrieved. `judge.py` scores the saved answers; an empty answer is scored 0.

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

### Generation results (DeepEval)

Judge: Claude Sonnet 4.6, a different model from the generator (`claude-haiku-5-5`). Metrics: **Faithfulness** (is the answer supported by the retrieved context?) and **Correctness** (does the answer match the hand-written expected answer, via G-Eval?). One run per setting.

| top-k | Faithfulness | Correctness |
| ----- | ------------ | ----------- |
| 5 | 0.954 | 0.807 |
| 10 | pending | pending |

K=10 scoring is pending (API credits exhausted). One K=10 response (question: What major regulation changes affected the 2022 Formula One season?) came back empty; it was not regenerated and will be scored 0 in the "all 27" figures and excluded from a like-for-like comparison over the same 26 questions at both settings.

Single runs: score differences under about 0.03 should be treated as noise. Correctness is a conservative lower bound, because some expected answers cover more than the question asks.

### What the failures show

- **Context pollution:** the "who won the 2023 Drivers' Championship" answer got the winner right but mixed in 2022 figures (454 points, Leclerc as runner-up) and called the correct 575 points "not reconciled". Faithfulness can stay high while correctness drops.
- **Redline artifact:** the 2023 floor-edge load test answer read a redlined passage's old and new values ("8mm 5mm") as an "8mm ±5mm" tolerance. This is the redline weakness described below.
- **Chunk coverage:** the Testing of Current Cars answer stated that its context was incomplete, because the article spans several chunks and top-5 did not retrieve all of them.

### Generation model

Answers are generated with the model set in the `ANTHROPIC_MODEL` environment variable (default `claude-haiku-5-5`).

## Known limitations

Some of the regulation PDFs use FIA's redline/revision-tracking convention (9 out of 16 of them), which causes some numeric values to appear duplicated in extracted text for recently-changed passages. Answers touching those passages should be treated as lower-confidence. The 2023 floor-edge load test question is anchored to redlined text and is answered incorrectly.

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

Python · FastAPI · LangChain · Anthropic Claude · Voyage AI · Qdrant · DeepEval (evaluation only) · Docker · uv