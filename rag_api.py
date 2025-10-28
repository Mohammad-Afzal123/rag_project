from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = FastAPI(title="RAG Query API", version="2.1")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

# Embeddings + Vectorstore
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

def retrieve_documents(query: str, k: int):
    return db.similarity_search(query, k=k)

# FLAN-T5 local pipeline
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    try:
        docs = retrieve_documents(req.query, req.top_k)
        contexts = [d.page_content.strip() for d in docs]
        combined_context = "\n\n".join(contexts)

        # Prepare prompt manually
        prompt = f"""
You are a medical assistant. Use the provided context to answer the question accurately and concisely.

Context:
{combined_context}

Question:
{req.query}

Answer:
"""
        # Generate answer
        output = flan_pipeline(prompt, max_new_tokens=512)
        answer = output[0]["generated_text"].strip()

        return QueryResponse(answer=answer, contexts=contexts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
