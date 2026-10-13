## 5.1 查询构建

### Text-to-SQL，自然语言转化为适用于关系数据库的SQL查询



```Python
user_query = "xxxx"

prompt = f"""
	数据库结构描述
    {schemna_desc}

    用户查询
    {user_query}

    返回SQL查询语句，不要包含任何解释、注释或格式标记
  
"""

response = llm.invoke(prompt)
```



#### 结合agent, sql-data-function和tool calls的方案


### Text-to-Cypher， 自然语言转化为适用于图数据库(Neo4j)的查询语言

## 5.2 查询翻译

### 使用Prompt重写查询

```Markdown
Generate search query for elasticsearch vector store from a user question, 
allowing for a more accurate response through semantic search.
Just return the revised elastic search query

user qquestion: {user_query}
revised elasticsearch query:
```



> refernces:
>
> 1. [www.elastic.co/search-labs/blog/query-rewriting-llm-search-improve](https://www.elastic.co/search-labs/blog/query-rewriting-llm-search-improve)
> 2. [docs.langchain.com/oss/python/integrations/retrievers](https://docs.langchain.com/oss/python/integrations/retrievers)


### 查询分解 query decomposition 

```Python
"""
请针对以下问题生成 3 个不同角度的检索查询。
保留原问题的核心意图，使用不同的关键词或表达方式。
每行输出一个查询，不要添加编号或解释。

原始问题：{question}
"""
```



### 查询澄清 - 逐步细化和明确用户的问题



> open ai guide
>
>
> 混合式问题澄清
> 在企业级 RAG 中，可以把澄清设计成一个独立的 Query Understanding / Query Refinement 阶段。
>
> 1. 用户原始问题 + 对话上下文
> 2. LLM 意图识别与信息抽取
>    识别问题类型、领域、实体、约束、时间范围及缺失参数
> 3. 领域模型与规则校验
>    匹配领域节点，检查必填条件、互斥条件和业务规则
>
> 信息充分
> 进入检索
>
> 关键条件缺失
> 向用户澄清
>
> 可直接检索
> 宽泛检索后判断
>
> 4. Hybrid Search → Rerank → 生成答案
>
> 其中，第三种分支很重要：不是所有信息缺失都值得向用户追问。
> 例如，用户询问“公司的差旅报销流程”，即使没有指定国内还是境外，系统也可以先检索通用流程。如果答案确实取决于差旅类型，再追问用户。这样可以减少不必要的对话轮次。
> 四、如何判断哪些问题必须澄清？
> 建议不要只依据 LLM 的主观判断，而是采用三类规则。
> 判断条件	处理策略
> 缺少回答所必需的参数	必须澄清
> 不同解释会导致实质不同的答案	优先澄清，或并行检索多个分支
> 信息不完整，但不影响主要结论	直接回答，并说明适用范围或假设
> 可以再引入两个指标：
>
> - 澄清成本：用户多回答一次问题的时间和交互成本。
> - 错误答案风险：不澄清就回答，导致答案错误或误导的风险。
>   当错误风险高于澄清成本时，优先澄清；否则可以尝试检索、回答或明确说明假设。这是一个策略框架，而不是必须采用固定数值的公式。
>   五、领域问题树应该如何构建？
>   我不建议人工为每一种用户问法建立树，而是定义以下内容：
>
> 1. 领域与意图分类：用户到底想完成什么任务。
> 2. 关键实体和参数：例如地区、产品、日期、费用类别。
> 3. 条件依赖关系：哪些参数只在特定情况下才需要。
> 4. 必需信息与可选信息：明确哪些缺失会阻止回答。
> 5. 知识源映射：不同问题类型对应哪些文档、索引或检索策略。
> 6. 澄清策略：优先询问什么、是否允许多选、是否可以带默认值。
>    领域树负责约束，LLM 负责理解与表达，检索结果则可以反过来帮助判断是否还需要澄清。
>
> 我的建议是：先用 LLM 做通用的问题理解，再逐步为高频、高风险的业务领域增加结构化领域模型和规则。
> 这样不需要一开始就投入大量精力构建完整的领域树，也能逐渐将关键业务场景从依赖模型判断，转变为可测试、可审计的确定性流程。



### 查询扩展

HyDE, hypothetical document emebdding

对于给定的查询，首先创建一个假设的回答文档，随后对该文档进行嵌入处理，并利用此嵌入做为最终查询的表示形式。

## 5.3 查询路由

### 逻辑路由

通过LLM对问题进行解析和语义理解，并根据问题的特点进行分类，生成结构化的输出以指导检索器的选择

### 语义路由

根据用户的语义，调用有针对性的提示模板。

> openai guide
>
>
> 最简单的实现：Embedding + 相似度阈值
> 语义路由不一定需要 LLM。一个简单、实用的实现只需要：
>
> 1. 为每个路由定义若干代表性问题。
> 2. 使用 Embedding 模型把这些问题转换成向量。
> 3. 在启动时计算并保存这些路由向量。
> 4. 运行时将用户问题向量化，计算相似度。
> 5. 选择最匹配的路由；如果相似度不足，则走默认路径。
>    这也是专门的语义路由库 semantic-router 所采用的基本思路之一。
>
> GitHub: [github.com/aurelio-labs/semantic-router/blob/main/docs/get-started/quickstart.md](https://github.com/aurelio-labs/semantic-router/blob/main/docs/get-started/quickstart.md)
>
