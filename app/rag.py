import os
from dotenv import load_dotenv
from typing import List, Optional
from qdrant_client import QdrantClient, models
import voyageai
from langchain_anthropic import ChatAnthropic
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
voyageai_client = voyageai.Client(api_key=os.getenv("VOYAGEAI_API_KEY"))
llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), max_tokens=1024)

collection_name = "motorsports_corpus"
voyage_model = "voyage-4"

SYSTEM_PROMPT = (
    "You are a Formula 1 regulations and history assistant. Answer using ONLY the "
    "provided context. Cite sources inline using [1], [2], etc., matching the "
    "numbered context blocks. If the context doesn't contain the answer, say so "
    "explicitly rather than guessing."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\nQuestion: {question}"),
])
chain = prompt | llm | StrOutputParser()


class QdrantMotorsportsRetriever(BaseRetriever):
    top_k: int = 5
    doc_type_filter: Optional[str] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vec = voyageai_client.embed(
            [query], model=voyage_model, input_type="query"
        ).embeddings[0]

        query_filter = None
        if self.doc_type_filter:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="doc_type", match=models.MatchValue(value=self.doc_type_filter)
                )]
            )

        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=self.top_k,
        )
        return [
            Document(
                page_content=p.payload["text"],
                metadata={
                    "source_url": p.payload["source_url"],
                    "section": p.payload["section"],
                    "doc_type": p.payload["doc_type"],
                },
            )
            for p in results.points
        ]


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] Source: {d.metadata['source_url']} ({d.metadata['doc_type']})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def answer_question(question: str, top_k: int = 5) -> dict:
    retriever = QdrantMotorsportsRetriever(top_k=top_k)
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = chain.invoke({"context": context, "question": question})
    sources = [
        {"n": i + 1, "url": d.metadata["source_url"], "doc_type": d.metadata["doc_type"]}
        for i, d in enumerate(docs)
    ]
    return {"answer": answer, "sources": sources}