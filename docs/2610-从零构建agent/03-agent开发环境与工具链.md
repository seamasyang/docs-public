
## 开发环境搭建

1. 安装python
2. 创建虚拟环境, python -m venv .venv
3. 依赖管理， requirements.txt

### 核心依赖

1. langchain & langgraph， 构建agent的核心逻辑和工作流;
2. langchain mcp adapter ->[upgrade] -> langchain.mcp
3. fastapi, 为构建API而设计的现代Python Web框架
4. uvicorn, 驱动异步python应用的高性能ASGI(async server gateway interface)服务器
5. openai python client, 通往先进语言智能的官方通道
6. python-dotenv, 管理应用配置与密钥的实用工具
7. redis, agent系统中的高速缓存与多功能数据中枢（会话管理、会话历史记忆），以及任务队列。
8. mysql-connector-python, 结构化数据持久化的基石
9. pydantic, agent数据流动的类型安全与结构正确
10. streamlit agent能力转化为交互式web应用
11. playwright， 赋予agent与现在web世界交互的能力
12. arxiv， agent与前沿科学文献的知识桥梁
13. sentence-transformers. NL->Vector
