from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio, traceback
from pydantic import BaseModel
from agent import safe_run_agent
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def noop_middleware(request: Request, call_next):
    return await call_next(request)

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=25.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent execution took too long")

class QueryResponse(BaseModel):
    model_output: str
    model_thoughts: list

@app.get("/run-agent", response_model=QueryResponse)
async def run_agent(query: str = Query(...)):
    try:
        result = await run_in_threadpool(safe_run_agent, query)
        output = result.get("output", "")
        thoughts = result.get("intermediate_steps", [])

        parsed = [
            {"thought": f"Used {act.tool} with input {act.tool_input}", "result": obs}
            for act, obs in thoughts
        ]

        return QueryResponse(model_output=output, model_thoughts=parsed)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
