from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from model import generate_response
import asyncio
import threading
import logging

logging.basicConfig(level=logging.ERROR)
app = FastAPI()

query_lock = threading.Lock()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query for the chatbot")

async def generate_response_with_timeout(query: str,max_tokens:int):
    try:
        return await asyncio.wait_for(asyncio.to_thread(generate_response, query, max_tokens), timeout=10)
    except asyncio.TimeoutError:
        return "Sorry, that took longer than expected!"

@app.post("/ask")
async def ask_chatbot(request: QueryRequest, max_tokens: int = 500):
    if not query_lock.acquire(blocking=False):  # Prevent multiple queries
        raise HTTPException(status_code=429, detail="Only one query can be processed at a time")
    try:
        response = await generate_response_with_timeout(request.query,max_tokens)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        query_lock.release()  # Release lock after processing

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
