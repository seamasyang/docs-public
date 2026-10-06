import os
from sentence_transformers import SentenceTransformer


slots = [
    "今天上海气如何",
    "请告诉我上海今天的天气",
    "上海天气预报",
    "什么是人工智能",
    "AI与人工智能有什么区别",
    "人工智能发展简史",
    "人工智能与机器学习的关系",
    "央视曝光缅北电诈园区地下靶场",
    "缅北电诈用AK47射击逃跑人员",
    "20多名缅北电诈人员集体出逃",
    "国庆旅游消费多点开花",
    "DeepSeek模型规模冲向10万亿参数",
    "华为与高通达成广泛专利许可协议",
    "香港三季度融资475亿美元创纪录",
    "贵州茅台i茅台APP暂停营业一天",
    "2026国庆档电影总票房突破9亿元",
    "郑钦文晋级中网女单16强",
    "西班牙首相宣布提前举行议会选举",
    "一油轮在霍尔木兹海峡遇袭后起火"
]


model_path = os.path.expanduser("~/.modelscope/Qwen3-Embedding-0.6B")
model = SentenceTransformer(model_path)

embeddings = model.encode(slots, normalize_embeddings=True)
print(embeddings)

# cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings)

# cluster mean
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
closet_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
df = pd.DataFrame({
    "slot": slots,
    "kmean labels": kmeans.labels_
})

for idx in closet_indices:
    print("-", slots[idx])

print(df.to_string(index=False))
