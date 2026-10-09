from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import uuid
import asyncio
import aiohttp
import json

app = FastAPI(title="A2A http sample")

#  "成都": "101270101"
# weibo 热搜

task_store = {}


@app.post("/rpc")
async def handle_request(request: Request):
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})    
    task_id = uuid.uuid4().hex

    # register task
    task_store[task_id] = {
        "method": method,
        "params": params,
        "status": "pending",
        "result": None,
    }

    asyncio.create_task(process_task(task_id))

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,            
            "result": {"task_id": task_id},
        }
    )


@app.get("/events/{task_id}")
async def stream_task(task_id):
    if not task_id or task_id not in task_store:
       return {"event": "error", "data":(f"task not found; id: {task_id}")}

    async def event_generator():
        while task_store[task_id]["status"] != "completed":
            yield {"event": "status", "data": f"{task_store[task_id]["status"]}"}
            await asyncio.sleep(3)

        yield {"event": "result", "data": f"{task_store[task_id]["result"]}"}

    return EventSourceResponse(event_generator())


async def process_task(task_id):

    agent_task = task_store[task_id]
    if not agent_task:
        raise ValueError(f"task not found; id: {task_id}")

    method_name = agent_task.get("method").lower()
    if method_name == "weibo":
        result = await _get_weibo()
    elif method_name == "weather":
        result = await _get_weather()
    else:
        raise ValueError(f"method not supported; id: {task_id}")
    
    task_store[task_id]["status"] = "completed"
    task_store[task_id]["result"] = result

async def _get_weibo():    
    url = "https://weibo.cn/pub/"    
    return await _get_request(url)

async def _get_weather(city_code:str = "101270101"):
    # ref: https://github.com/zaixiamaomaoyu/weather-china-mcp/blob/master/city-codes-complete.json
    url = f"https://www.weather.com.cn/weather1d/{city_code}.shtml"
    headers = {
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
           "Referer": "http://www.weather.com.cn/",
       }
    text =  await _get_request(url, headers)
    print(f"weibo resposne: {text}")
   
    return text

async def _get_request(url, headers=None):
    if headers == None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url=url) as response:
            return await response.text()

# command to start it 
# uvicorn 01-03-A2A-http:app --host 0.0.0.0 --port 8280 --reload