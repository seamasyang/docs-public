import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

original_slots = [
    "上海今天天晴，气温20度，空气质量好",
    "深圳今天多云，气温30度，零星小雨",
    "用户希望能对这两个城市的天气进行进阶总结",
    "生成的内容比较包含城市，气温与天气情况"
]
summary_slot="上海天晴、气温20度，深圳则为零星小雨且气温30度。用户希望简明说明天气。"

model_path = os.path.expanduser("~/.modelscope/Qwen3-Embedding-0.6B")
model = SentenceTransformer(model_path)
original_embeddings = model.encode(original_slots)
summary_embeddings = model.encode([summary_slot])

similarities  = cosine_similarity(original_embeddings, summary_embeddings).flatten()
print(similarities)

semantic_mean_score = np.mean(similarities)
print(semantic_mean_score)

stopwords=[]
stopwords_path = f"{os.path.dirname(os.path.abspath(__file__))}/baidu_stopwords.txt"
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

import jieba
corpus = [" ".join(jieba.lcut(text)) for text in original_slots]
vectorizer = TfidfVectorizer( stop_words=stopwords) # ngram_range=(1,2),
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = np.array(vectorizer.get_feature_names_out())

print(tfidf_matrix)
print(feature_names)

# get the top indices
avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top_indices = avg_tfidf.argsort()[-10:][::-1]
key_phases = feature_names[top_indices]
print(top_indices)
print(key_phases)

# coverage 
matched_keywords = [kw for kw in key_phases if kw in summary_slot]
coverage_ratio = len(matched_keywords) / len(key_phases)
print(matched_keywords)
print(coverage_ratio)