from pydantic import BaseModel, Field
from typing import List

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class SourceItem(BaseModel):
    n: int
    url: str
    doc_type: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem]