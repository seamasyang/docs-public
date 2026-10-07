from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import uuid
import asyncio
import aiohttp
import json
import time
import logging
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

app = FastAPI(title="A2A demo")
logger = logging.getLogger()

class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed="completed"
    failed="failed"

class AgentTask(BaseModel):
    task_id:str
    sender:str
    receiver:str
    payload:dict
    status: TaskStatus = TaskStatus.pending
    result: Optional[dict] = None
    error: Optional[str] = None

class TaskCreation(BaseModel):
    sender:str
    receiver: str
    payload: dict 


task_store: dict[str, AgentTask] = {}


async def execute_task(task_id:str):
    task = task_store.get(task_id)
    task.status = TaskStatus.running
    logger.debug(f"[running] task {task_id}; {task.sender}->{task.receiver}")

    await asyncio.sleep(10)

    result = {"echo":task.payload.get("input", "")}
    task.status = TaskStatus.completed
    task.result = result
    logger.debug(f"[success] task {task_id} completed; {result}")



@app.post("/task")
async def create_task(request: TaskCreation, background_tasks: BackgroundTasks):
    task_id = uuid.uuid4().hex

    # register task
    task_store[task_id] = AgentTask(
        task_id=task_id,
        sender=request.sender,
        receiver=request.receiver,
        payload=request.payload
    )
    background_tasks.add_task(execute_task, task_id)

    return JSONResponse(
        {
            "task_id": task_id,
            "status":task_store[task_id].status
        }
    )


@app.get("/task/{task_id}")
def get_task(task_id:str):    
    if not task_id or task_id not in task_store:
       return {"event": "error", "data":(f"task not found; id: {task_id}")}       

    async def event_generator():
            current_task = task_store[task_id] 
            while current_task.status != TaskStatus.completed:
                yield {"event": "status", "data": current_task.status}
                await asyncio.sleep(2)
    
            yield {"event": "result", "data": current_task.model_dump()}
    
    return EventSourceResponse(event_generator())




# command to start it 
# uvicorn 01-06-A2A-task:app --host 0.0.0.0 --port 8280 --reload