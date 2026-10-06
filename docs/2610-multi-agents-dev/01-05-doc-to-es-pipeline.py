import os
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings

from docling.datamodel.pipeline_options import PipelineOptions
from docling.document_converter import DocumentConverter, MarkdownFormatOption
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from langchain_text_splitters import MarkdownHeaderTextSplitter

from transformers import AutoTokenizer



embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b", base_url="http://localhost:11434", dimensions=768
)
vector_store = ElasticsearchStore(
    index_name="app-demo",
    embedding=embeddings,
    es_url="http://192.168.1.7:9200",
)
model_path = os.path.expanduser("~/.modelscope/Qwen3-Embedding-0.6B")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, padding_side="left"
)


file_path = f"{os.path.dirname(os.path.abspath(__file__))}/2610-multi-agents-dev/01-协议与框架设计基础.md"
pipeline_options = PipelineOptions()
pipeline_options.allow_external_plugins = True
converter = DocumentConverter(
    format_options={
        "markdown": MarkdownFormatOption(pipeline_options=pipeline_options)
    }
)
loader = DoclingLoader(
    file_path=file_path,
    converter = converter,
    export_type=ExportType.MARKDOWN,
    chunker=HybridChunker(tokenizer=tokenizer, max_tokens=8192),
)
docs = loader.load()
for doc in docs:
    print(doc.page_content)

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
    ],
)
splitted_docs = [
    split for doc in docs for split in splitter.split_text(doc.page_content)
]

vector_store.add_documents(splitted_docs)
