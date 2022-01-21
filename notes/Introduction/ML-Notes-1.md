---
title: "Introduction"
date: 2022-01-20 19:54:49
categories:
  - "机器学习"
---

对概率的诠释有两大学派，一种是频率派，一种是贝叶斯派。
$$
X_{N\times p}=(X_{1}, X_{2}, \cdots, X_{N})^{T},X_{i}=
(X_{i1}, X_{i2}, \cdots, X_{ip})^{T}
$$
上述表达式中，表示有*N*个样本，每个样本都是*p*维向量。其中每个观测都是由$p(x|\theta)$生成的。

# 频率派

$p(x|\theta)$中，$\theta$是一个常量。对于$N$个观测来说，观测集的概率为$p(X|\theta)\mathop{=}\limits_{i_{id}}\prod_{i-1}^{N}p(x_{i}|\theta)$。

为了求$\theta$，采用最大对数似然估计MLE方法：
$$
\theta_{MLE}=\mathop{argmax}\limits_{\theta}\log{p(X|\theta)}\mathop{=}\limits_{i_{id}}\mathop{argmax}\limits_{\theta}\sum_{i=1}^{N}{\log{p(x_i|\theta)}}
$$

## 贝叶斯派

贝叶斯派认为$p(x|\theta)$中的$\theta$不是一个常量。这个$\theta$满足一个预设的先验分布$\theta\sim p(\theta)$。

根据贝叶斯定理依赖观测集参数的后验概率就可以写成:
$$
p(\theta|X)=\frac{p(X|\theta)\cdot p(\theta)}{p(X)}=\frac{p(X|\theta)\cdot p(\theta)}{\int\limits_{\theta}{p(\theta)d\theta}}
$$
为了求$\theta$的值，我们使用最大后验概率MAP：
$$
\theta_{MAP}=\mathop{argmax}\limits _{\theta}p(\theta|X)=\mathop{argmax}\limits _{\theta}p(X|\theta)\cdot p(\theta)
$$
由于分母和$\theta$没有关系。求解这个$\theta$值后计算$\frac{p(X|\theta)\cdot p(\theta)}{\int\limits _{\theta}p(X|\theta)\cdot p(\theta)d\theta}$，就得到了参数的后验概率。其中$\p(X|\theta)$叫做似然，是模型分布。得到了参数的后验分布后，我们就可以将这个分布用作预测贝叶斯预测：
$$
p(x_{new}|X)=\int\limits _{\theta}p(x_{new}|\theta)\cdot p(\theta|X)d\theta
$$
其中积分中的被乘数是模型，乘数是后验分布。
