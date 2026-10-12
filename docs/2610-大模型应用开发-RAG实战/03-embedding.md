## 3.1 嵌入是外部信息的编码

如何计算向量相似度？

**欧几里得距离**： 两个点之间的直线距离

$$
d^2=\sum_i(x_i-y_i)^2
$$

如果是二维空间

$$

d^2={(x_2-x_1)^2+(y_2-y_1)^2}
$$


**曼哈顿距离**


$$
d(X,Y) = \sum_i |x_i - y_i|
$$

如果是二维空间

$$
d = |x_2 - x_1| + |y_2 - y_1|
$$


**余弦相似度**

$$

\operatorname{cosine\_similarity}(X,Y)
=
\frac{X \cdot Y}{\|X\|\|Y\|}
=
\frac{\sum_i x_i y_i}
{\sqrt{\sum_i x_i^2}\sqrt{\sum_i y_i^2}}
$$

二维空间

$$
\operatorname{CosineSimilarity}(A,B)
=
\frac{x_1x_2+y_1y_2}
{\sqrt{x_1^2+y_1^2}\sqrt{x_2^2+y_2^2}}
$$


**点积**

$$
A \cdot B = x_1x_2 + y_1y_2
$$



## 3.2 词嵌入到大模型嵌入

## 3.3 现代嵌入模型



> MTEB, 海量文本嵌入基准测试
>
> 1. 检索， 信息检索BEIR
> 2. 文本匹配，语义相似度STS-B
> 3. 分类，句子分类(Amazon Reviews)
> 4. 聚类，主题聚类(TREC))
> 5. 排序，语义排名(MS MARCO)
> 6. 摘要评估，摘要质量评估(SummEval)
> 7. 回归， 句子回归(STS-B)
> 8. 生成任务辅助，评估输入输出质量
>



## 3.4 稀疏嵌入、密集嵌入和BM25

## 3.5 多模态嵌入

## 3.6 使用LC, LlamaIndex框架

## 3.7 微调嵌入模型
