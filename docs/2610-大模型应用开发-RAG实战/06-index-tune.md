## 6.1 从小到大：节点-句子滑动窗口和父子文本块

langchain, chila-parent recursive retrieval

### 节点-句子滑动窗口

### 父子文本块检索： 小块搜索，大块返回。

原始文档经过预处理后形成子文本块(子文档)被存入向量库, 原始wendang(父文档）也会被保存。

当查询时，从向量库找到相关片段，然后利用文档存储获取这些片段所属的完整类容。

> openai guide
>
>
> 原始文档 
> -> 
>
> 切分父块（Parent Chunk）
>
> 例如 1,000 字符
>
> -> 
>
> 进一步切分子块（Child Chunk）
>
> 例如 250 字符，每个子块独立生成向量
>
> ->
>
> `<span>parent_docs</span>`
>
> 父块原文，不需要向量
>
> `<span>child_chunks</span>`
>
> 子块原文 + 1024 维向量 + 父块 ID
>
> ->
> 子块 kNN 检索 → 父块去重 → Elasticsearch 批量获取父块



```Python
# 示例参数；应根据文档类型与评测结果调整
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
```


## 6.2 粗中有细： 利用IndexNode和RecursiveRetrieve构建从摘要到细节的索引

Summary-to-details： 先整理高层信息，利用层次化索引和递归检索的方式，是检索和生成可按需逐步深入

粗粒度：主索引包含概述行摘要信息，可快速定位目标场景或主题

细粒度：当需要更详细的信息时，可以通过与主索引相连的二级索引（或更深)递归式地检索细节内容


## 6.3 分层合并： HierachicalNodeParser和PARTOR

先将文本分成多个层次，随后进行索引，形成多种向量表示，最后检索过程对其整合


利用PARTOR递归生成多层级索引


## 6.4 前后串连




## 6.5 混合检索

Ensemble Retriever结合BM25和语义检索


Ensemble Retriever将两种检索的结果组合起来，通过混合检索策(rrf)略整合结果，并最终传递给生成器以输出结果。
