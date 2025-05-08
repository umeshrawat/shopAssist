from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    vectorType: str = "faiss"
    k: int = 2

class SearchResponse(BaseModel):
    result: str
    url: str