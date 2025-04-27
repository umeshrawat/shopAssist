from fastapi import FastAPI
from src.retriever.retriever import Retriever
from src.generator.response_generator import Generator

app = FastAPI()

retriever = Retriever()
generator = Generator()

@app.get("/")
def read_root():
    return {"message": "Shopping Assistant RAG API is running."}

@app.post("/query")
def query_shopping_assistant(question: str):
    retrieved_docs = retriever.retrieve(question)
    response = generator.generate(retrieved_docs, question)
    return {"response": response}
