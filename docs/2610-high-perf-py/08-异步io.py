import asyncio

# 定义一个协程
async def fetch_data():
    print("1. 开始请求网络...")
    # 模拟网络 I/O：遇到 await，当前协程暂停，交出控制权！
    await asyncio.sleep(2) 
    print("2. 网络请求完成，继续执行...")
    return "数据"

# 调用协程函数，会返回一个协程对象（不会立刻执行）
coro = fetch_data()


# 模块顶层（同步环境）
# !! 不能使用await coro
if __name__ == "__main__":
    asyncio.run(coro)   # 启动异步程序