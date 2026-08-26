from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import QuestionRequest, AnswerResponse
from app import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag.qdrant_client.get_collection(rag.collection_name)
    except Exception as e:
        raise RuntimeError(f"Qdrant collection '{rag.collection_name}' not reachable: {e}")
    yield


app = FastAPI(
    title="F1 Regulations & History Q&A",
    description="RAG service answering Formula 1 regulation and history questions with cited sources.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    try:
        rag.qdrant_client.get_collection(rag.collection_name)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        return rag.answer_question(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"service": "F1 Regulations and History Q&A", "docs": "/docs"}