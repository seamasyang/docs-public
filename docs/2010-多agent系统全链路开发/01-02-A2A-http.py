from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import uuid
import asyncio

app = FastAPI(title="A2A http sample")

task_store = {}


@app.post("/rpc")
async def handle_request(request: Request):
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    session_id = uuid.uuid4().hex
    task_id = uuid.uuid4().hex

    # register task
    task_store[task_id] = {
        "method": method,
        "params": params,
        "status": "pending",
        "result": None,
    }

    asyncio.create_task(simulate_task(task_id, params))

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": session_id,
            "result": {"task_id": task_id},
        }
    )


@app.get("/events/{task_id}")
async def stream_task(task_id):
    async def event_generator():
        while task_id in task_store and task_store["task_id"]["status"] != "completed":
            yield {"event": "status", "data": f"{task_store[task_id]["status"]}"}
            await asyncio.sleep(0.5)

        yield {"event": "result", "data": f"{task_store[task_id]["result"]}"}

    return EventSourceResponse(event_generator())


async def simulate_task(task_id, params):
    await asyncio.sleep(5)
    input_text = params.get("text", "")
    result = input_text.upper()
    task_store[task_id]["status"] = "completed"
    task_store[task_id]["result"] = result
