from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool
import asyncio
import os


@tool
async def tool_a():
    "append prefix A to input text"
    await asyncio.sleep(2)
    return f"[A-async]{input}"


@tool
async def tool_b():
    "append suffix B to input text"

    await asyncio.sleep(2)
    return f"{input}[B-async]"


async def run_agent_chain():
    tools = [tool_a, tool_b]    

    llm = ChatOpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        model="deepseek-flash",
    )

    agent = create_agent(model=llm, tools=tools)
    response = await agent.ainvoke(input={"messages": [HumanMessage(content="Transform 'hello world' by adding prefix A and suffix B")]})
    messages = response.get("messages", [])
    print(messages[-1])


asyncio.run(run_agent_chain())
