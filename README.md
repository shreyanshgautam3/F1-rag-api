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

- **Chunking** — Wikipedia articles are split on their headers, then into 500-token chunks (80 overlap) for longer sections; regulation PDFs have repeated headers/footers stripped, then are split into 280-token chunks (40 overlap) (`langchain-text-splitters`, `tiktoken`)
- **Embeddings** — Voyage AI
- **Vector store** — Qdrant Cloud
- **Retrieval + generation** — a custom LangChain `BaseRetriever` wrapping the Qdrant/Voyage calls, composed via LCEL with `ChatAnthropic` (Claude) and a `StrOutputParser`
- **Observability** — Langfuse tracing
- **Evaluation** — golden-set retrieval metrics, DeepEval generation metrics, GitHub Actions retrieval check
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

`eval.py` takes the top-k as an argument (default 5). For each question it retrieves the top-k chunks, generates an answer from those same chunks, and saves the question, expected answer, generated answer, retrieved contexts, source URLs and rank of the first correct source. It prints Hit Rate and MRR and lists any questions whose expected source was not retrieved. Adding `--retrieval-only` skips answer generation, writes no results file, and exits with an error if Hit Rate falls below 0.96 (this is what CI runs). `judge.py` scores the saved answers; an empty answer is scored 0.

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

Judge: Claude Sonnet 4.6 via DeepEval (≥ 4.3.5), a different model from the generator (`claude-haiku-5-5`). Metrics: **Faithfulness** (is the answer supported by the retrieved context?) and **Correctness** (does the answer match the hand-written expected answer, via G-Eval?). One run per setting.

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

### Continuous integration

![Eval](https://github.com/shreyanshgautam3/F1-rag-api/actions/workflows/eval.yml/badge.svg)

A GitHub Actions workflow (`.github/workflows/eval.yml`) runs on pushes that touch `app/`, `eval.py`, `golden_set.json` or the dependency files:

- **Retrieval check (automatic):** runs the golden set at k=5 with no LLM calls and fails the build if Hit Rate drops below 0.96 (one miss out of 27 allowed). MRR is not gated because it varies by about ±0.012 between runs.
- **Generation + judge (manual only):** "Run workflow" with `run_judge` ticked generates answers and scores them with DeepEval. It is manual because it uses Anthropic credits.

Repository secrets: `QDRANT_URL`, `QDRANT_API_KEY`, `VOYAGEAI_API_KEY`, and `ANTHROPIC_API_KEY` (manual judge job only).

## Technical decisions

| Decision | Choice | Why | Trade-off |
| --- | --- | --- | --- |
| Chunking | Wikipedia: header split, then 500 tokens (80 overlap) for sections over 80 tokens. PDFs: boilerplate stripped, 280 tokens (40 overlap). 8,219 chunks total. | Wikipedia articles carry their own structure, so keeping sections whole preserves meaning. Extracted PDFs have no markup, and short chunks suit precise rule lookups. | Multi-part regulation articles span several chunks. Evaluation showed the Testing of Current Cars answer missing parts of its article at top-5. Token length uses the `cl100k_base` tokenizer as an approximation, not Voyage's own. |
| Embeddings | Voyage `voyage-4`, 1024 dimensions, cosine distance, `document` vs `query` input types | Retrieval-tuned embeddings, with the asymmetric query/document modes. And recommended by Anthropic. | A hosted API call per query. |
| Vector store | Qdrant Cloud, with `text`, `source_url`, `section` and `doc_type` in the payload and deterministic point IDs (hash of source + chunk index) | A managed service, so there is no database to run on Render. Stable IDs make re-ingestion upsert the same points. | A network hop on every query. The notebook ingestion step deletes and rebuilds the whole collection. |
| Retriever | A small custom LangChain `BaseRetriever` | Embedding, filtering and payload mapping stay explicit, and retrieval can be measured separately from generation, which `eval.py` relies on. | More code than a framework default. |
| Generation | Claude Haiku (`claude-haiku-5-5`, `max_tokens=1024`); the system prompt restricts answers to the context, requires `[n]` citations, and says to admit missing context | Cheap enough for an open endpoint. | A small model sometimes over-trusts mixed context (the 2022/2023 failure). |
| Default `top_k` | 5, with the API accepting 1-20 | Hit Rate is already 1.000 at k=3, and k=5 adds margin. The K=10 experiment is in the table below. | More chunks mean more tokens, and potentially more pollution from sibling documents. |
| Error handling | Pydantic bounds on input (422 before any model call), generic 500 and 503 messages, server-side logging, a startup check that the Qdrant collection is reachable | No internals leak to callers, and bad input never costs an LLM call. | Debugging requires reading server logs. |
| Evaluation | Hand-written golden set, retrieval metrics gated in CI, DeepEval judged manually with a judge model different from the generator | Retrieval metrics are almost free, so they run on every push. The judge costs credits, so it is manual. | Single runs per setting, so results are directional. |

### Experiment: top-k 5 vs 10

| top-k | Hit Rate | MRR | Faithfulness | Correctness |
| --- | --- | --- | --- | --- |
| 5 | 1.000 | ≈0.92 | 0.954 | 0.807 |
| 10 | 1.000 | ≈0.91 | pending | pending |

K = 10 generation scores are pending (see "Generation results" above); this comparison will be completed once they are available.

### What I would change next

- Split regulation PDFs by Article instead of fixed token windows, so multi-clause rules stay together.
- Store season and issue as metadata so queries can filter by year and avoid cross-season pollution.
- Add a reranker over the top-10 candidates.
- Detect and collapse redline text during PDF extraction.
- Pin Wikipedia revisions when scraping.
- Run each evaluation setting several times and report the mean.

## Known limitations

Some of the regulation PDFs use FIA's redline/revision-tracking convention (9 out of 16 of them), which causes some numeric values to appear duplicated in extracted text for recently-changed passages. Answers touching those passages should be treated as lower-confidence. The 2023 floor-edge load test question is anchored to redlined text and is answered incorrectly.

The corpus also contains several issues of the same regulations. Four golden-set questions miss at k=1 (all are recovered by k=3). They concern topics covered by multiple near-duplicate documents, such as several issues of the same regulation, so a sibling document can rank first.

## Threat model

Scope: a public, read-only RAG endpoint over a public corpus (FIA regulations and Wikipedia). The LLM has no tools and takes no actions, the endpoint has no authentication, and no secrets or private data are in the model's context. Risks are mapped to the OWASP Top 10 for LLM Applications (2025).

| Risk | How it applies here | In place | Not done |
| --- | --- | --- | --- |
| LLM01 Prompt injection | A user question can try to override instructions. Retrieved text from scraped sources can also contain instructions (indirect injection). | The system prompt restricts answers to the provided context. Questions are limited to 3-500 characters. The LLM has no tools, so the worst outcome is a bad answer, not an action. | Retrieved context is not delimited or labeled as untrusted data. No input/output filtering. Not yet scanned (Promptfoo scan planned). |
| LLM04 / LLM08 Poisoning and retrieval weaknesses | Wikipedia is editable, and sibling regulation issues and adjacent seasons compete in retrieval. Evaluation found 2022 figures leaking into a 2023 answer. | Retrieval is measured against a golden set (Hit Rate, MRR) and checked in CI. | No filtering by season or document issue, no reranking, no pinning of Wikipedia revisions. |
| LLM09 Misinformation | Evaluation showed a wrong 2023 answer (mixed-in 2022 points) and a redline-artifact error on a floor-test question. | Answers cite numbered sources returned in the response. The prompt requires saying so when the context lacks the answer. Faithfulness and correctness are tracked with DeepEval. | Known failures are documented, not fixed. |
| LLM10 Unbounded consumption | Each call costs a Voyage embedding and a Claude call. | Per-request bounds: `top_k` limited to 1-20, question limited to 500 characters, answer limited to 1024 tokens. | No authentication or per-client rate limiting in the application. |
| LLM02 Sensitive information disclosure | Error text could expose internals; questions are stored in tracing. | Responses contain only the answer and source URLs, never raw chunks. Errors return a generic message and are logged server-side. Secrets are environment variables. | Questions and answers are logged in Langfuse, so private data should not be submitted. |
| LLM06 Excessive agency | Not applicable: the pipeline is read-only retrieval plus text generation, with no tools. | n/a | n/a |
| LLM07 System prompt leakage | Low risk: the system prompt contains no secrets and the repository is public. | n/a | n/a |

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

Input limits: `question` must be 3-500 characters and `top_k` must be 1-20; requests outside these bounds are rejected with 422 before any model call. Unexpected failures return 500 with `{"detail": "Internal error"}`, and the full error is logged server-side only.

**`GET /health`** — returns `{"status": "ok"}` when the Qdrant collection is reachable, otherwise 503.

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

Python · FastAPI · LangChain · Anthropic Claude · Voyage AI · Qdrant · Langfuse · DeepEval (evaluation only) · GitHub Actions · Docker · uv