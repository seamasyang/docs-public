from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import uuid
import asyncio
import logging
from typing import Any, Callable
from dataclasses import dataclass
import time



@dataclass
class AgentTask:
    agent_id:str
    handler: Callable[[dict], Any]


class AgentManger:
    def __init__(self):
        self._agents={}

    def register(self, agent_id:str, handler: Callable[[dict], Any]):
        self._agents[agent_id] = AgentTask(agent_id=agent_id, handler=handler)

    def get(self, agent_id:str) -> Callable[[dict], Any]:
        if agent_id not in self._agents:
            raise ValueError(f"Agent not found; id: {agent_id}")

        return self._agents[agent_id].handler


async def dummy_agent_task(agent_id:str, payload:dict)->dict:
    await asyncio.sleep(5)

    return {
        "agent_id": agent_id,
        "input": payload,
        "output": f"task completed. agent id: {agent_id}, task: {payload.get('task')}",
        "updated_time":  time.time()
    }


logger = logging.getLogger()
app = FastAPI(title="A2A demo")

# register agent
agent_manager = AgentManger()
agent_manager.register("retrieve", dummy_agent_task)
agent_manager.register("summarize", dummy_agent_task)
agent_manager.register("translate", dummy_agent_task)
agent_manager.register("faq", dummy_agent_task)

agent_task_store = {} # id, task plan

async def run_tasks(task_id:str):
    logger.debug(f"prepare agent task; id: {task_id}")
    agent_tasks = []
    task_plan = agent_task_store[task_id]
    for agent_id, payload in task_plan.items():
        logger.debug(f"agent task started; id: {agent_id}")
        handler = agent_manager.get(agent_id)
        agent_task = _safe_execute(handler, agent_id, payload)  # Notes: no await; indeed await lies two lines afterwards
        agent_tasks.append(agent_task)
    results = await asyncio.gather(*agent_tasks) # return when all completed
    agent_task_store[task_id] = results # write result back
    return results

async def _safe_execute(handler, agent_id, payload):
    try:
        result = await handler(agent_id, payload)
        return {"agent_id":agent_id, "status": "completed", "result": result}
    except Exception as e:
        return {"agent_id":agent_id, "status": "failed", "error": str(e)}


@app.post("/task")
async def create_task(request: Request):

    task_id = uuid.uuid4().hex
    payload = await request.json()

    agent_task_store[task_id] = payload

    asyncio.create_task(run_tasks(task_id))

    return JSONResponse(
        {
            "task_id": task_id
        }
    )


@app.get("/task/{task_id}")
def get_task(task_id:str):
    if not task_id or task_id not in agent_task_store:
       return {"event": "error", "data":(f"task not found; id: {task_id}")}

    async def event_generator():
        while True:
            current_agent_tasks = agent_task_store[task_id]
            logger.debug(f"==============> current agent tasks: {current_agent_tasks}")
            if all(r["status"] in ("completed", "failed") for r in current_agent_tasks):
                break
            yield {"event": "status", "data": "processing"}
            await asyncio.sleep(2)

        yield {"event": "result", "data": current_agent_tasks}

    return EventSourceResponse(event_generator())
