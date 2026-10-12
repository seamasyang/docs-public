
## 2.1 为什么分块重要

## 2.2 固定字符分块

## 2.3 递归分块

## 2.4 基于特定格式分块

## 2.5 基于文件结构或语义分块

## 2.6 与分块相关的高级索引构建技巧

> LangChain -> ParentDocumentRetriever 在文档分块时，
>
> 首先将文档切分为较大的父文本块，
> 然后再进一步将这些父文本块细分为更小的子文本块。
> 每个子文本块都保留一个指向其父文本块的ID，从而确保信息的连贯性和可追溯性
>
> 在存储阶段，对子文本块生成嵌入向量并存入向量库，同事单独存储父文本块。这两个存储通过父文本块的ID进行关联。
>
> 检索阶段，系统会找到与查询最相关的子文本块
>
> 生成阶段，通过子文本块的ID定位相关的父文本，并将两者联合起来传递给大模型，从而更加丰富和准确的回答



		Original File
                         │
                         ▼
                  Document Parser
                         │
                         ▼
                Semantic Structure
                         │
                ┌────────┴────────┐
                ▼                 ▼
             Parent             Child
                │                 │
                │                 ▼
                │          Elasticsearch
                │          ┌─────────────┐
                │          │ BM25        │
                │          │ Vector      │
                │          │ parent_id   │
                │          └─────────────┘
                │                 │
                │                 ▼
                │             Reranker
                │                 │
                │                 ▼
                │              parent_id
                │                 │
                └─────────────────┘
                         │
                         ▼
                   Parent Store
                         │
                         ▼
                       LLM


ES Index
├── rag_parent
│     ├── parent_id
│     ├── content
│     └── metadata
│
└── rag_child
      ├── child_id
      ├── parent_id
      ├── content
      ├── embedding
      └── metadata
