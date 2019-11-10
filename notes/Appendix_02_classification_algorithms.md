# 分类模型算法推导的简要概述

## 1.Logistic回归
(1) <b>假设函数(hypothesis function)</b>：<br/>
把θ<sup>T</sup>.X作为sigmoid函数的参数，从而将值域从`(-∞,∞)`$映射到`(0,1)`，其中0映射到0.5 <br/>
(2) <b>似然函数(likelyhood function)</b>: <br/>
映射后H(θ, X) = sigmoid(θ<sup>T</sup>·X) 是样本为正例的预测概率, 根据样本标签Y<sub>i</sub>取值有1，0两种
> Y<sub>i</sub>=0 时使用 1 - H(θ, X) = 1 - sigmoid(θ<sup>T</sup>·X) 表示样本为负例的预测概率<br/>
> Y<sub>i</sub>=1 时使用 sigmoid(θ<sup>T</sup>·X) 表示样本为正例的预测概率

合并两个公式得到 P(Y<sub>i</sub>|X<sub>i</sub>, θ) = (H(θ,X<sub>i</sub>)^Y<sub>i</sub>) * (1 - H(θ,X<sub>i</sub>)^(1-Y<sub>i</sub>))，值取最大时预测最准确<br/>
带入所有样本，就可以得到似然函数L(θ)（样本和标签已知的条件下，表示模型参数θ取特定值的概率的函数）
> 公式：L(θ)= ∏(H(θ,X<sub>i</sub>)<sup>Y<sub>i</sub></sup>)*(1 - H(θ,X<sub>i</sub>))<sup>(1-Y<sub>i</sub>)</sup>)

(4) <b>对数似然函数</b>：对似然函数取对数得到对数似然函数
> log(L(θ))=∑(Y<sub>i</sub>\*log(H(θ,X<sub>i</sub>))+(1-Y<sub>i</sub>)\*log(1 - H(θ,X<sub>i</sub>)))

(5) <b>梯度上升求模型参数</b>：似然函数取最大值时，模型的精确度最高，进而可以用求偏导、梯度上升的方法求解 <br/>
备注1：H(θ,X)表示样本为正的预测概率, 是来自作者的设计思路
> 他想设计一个函数使得log (P<sub>positive</sub>/(1-P<sub>positive</sub>))是线性函数，进而能够用θ<sup>T</sup>·X表示，最终他使用了sigmoid函数。<br/>
> 推导过程: log (P/(1-P)) = log (H(θ,X)/(1 - H(θ,X)) = log (1 / e<sup>(-θ<sup>T</sup>·X)</sup>)</sup> = θ<sup>T</sup>·X

备注2：Logistic Regression 的损失函数，就是负的对数似然函数
> 对数似然函数（梯度上升）取到最大值 <=> 损失函数 (梯度下降）取到最小值

## 2.Softmax回归
K分类、第K类参数为为列向量θ<sub>k</sub>，所有类别的参数列向量可以组成矩阵θ<sub>k⨯n</sub><br/>
(1) 类别为K的概率密度: P<sub>k</sub>=P(c=k|X;θ)=e<sup>(θ<sub>k</sub><sup>T</sup>·X)</sup>/SUM(e<sup>θ<sub>i</sub><sup>T</sup>·X</sup>) </sup><br/>
(2) 然而softmax并没有直接使用P<sub>k</sub>/SUM(P<sub>i</sub>)建模，而是用 e<sup>P<sub>k</sub>/SUM(P<sub>i</sub>)</sup>来近似  
> 因为P<sub>i</sub> < P<sub>j</sub>时，e<sup>P<sub>i</sub></sup> < e<sup>P<sub>j</sub></sup>，且e<sup>P<sub>i</sub></sup> / SUM(e<sup>P<sub>i</sub></sup> + e<sup>P<sub>j</sub></sup>) < e<sup>P<sub>j</sub></sup> / SUM(e<sup>P<sub>i</sub></sup> + e<sup>P<sub>j</sub></sup>) <br/>
> 反之依然成立，不影响效果但是方便推导

(3) 进一步，用 ∏<sub>k</sub>P(c=k|X<sup>i</sup>,θ)<sup>y<sub>k</sub></sup> 可以表示样本i属于类别k的概率<br/>

(4)	假设样本间相互独立，可以得到模型的似然函数
> L(θ) = ∏<sub>i</sub>∏<sub>k</sub>(P(c=k|X<sub>i</sub>,θ)<sup>y<sub>k</sub></sup>) = ∏<sub>i</sub>∏<sub>k</sub>((e<sup>P<sub>k</sub></sup>/SUM(e<sup> P<sub>i</sub></sup>))<sup>Y<sub>k,i</sub></sup>))<br/>

模型的对数似然函数
> log(L(θ)) = SUM<sub>i</sub>(SUM<sub>j</sub>(Y<sub>k,i</sub>*(P<sub>k</sub> - log(SUM(e<sup>P<sub>j</sub></sup>))))

(5)	对这个对数似然函数求偏导，做梯度上升，得到使似然函数取最大值时的参数θ

备注:  k=2时，令θ=θ<sub>1</sub>-θ<sub>2</sub>，Softmax回归退化为Logistic Regression</br>
> P<sub>1</sub> = e<sup>θ<sub>1</sub><sup>T</sup>·X</sup> / (e<sup>θ<sub>1</sub><sup>T</sup>·X</sup> + e<sup>θ<sub>2</sub><sup>T</sup>·X</sup>) = 1 / (1 + e<sup>-(θ<sub>1</sub>-θ<sub>2</sub>)<sup>T</sup>·X</sup>) = 1/(1+e<sup>-θ<sup>T</sup>·X</sup>)

#3.Logistic Regression/Softmax与相对熵
(1) 相对熵（也叫交叉熵、鉴别信息、Kullback熵，Kullback-Leible散度，K-L距离）</br>
> 若p(x)是我们在研究的概率分布（例如样本概率），q(x)为一个基准分布（例如似然概率）</br>
> 则p(x)与q(X)的相对熵D(p||q)，可以理解为对log(P(x)/Q(x))求数学期望，即
D(p||q) = SUM(p(x)*log(p(x)/q(x))) 

(2) 在logistic regression中，`p(x)`被建模为样本类别的真实概率分布，`q(x)`被建模为似然概率。即：
> p(X<sub>i</sub>) = Y<sub>i</sub></br>
> q(X<sub>i</sub>) = sigmod(θ<sup>T</sup>·X) = 1 / (1 + e<sup>(-1 * sigmod(θ<sup>T</sup>·X))</sup>)</br>

带入到相对熵公式，可得到
> D(p||q) = SUM(p(X<sub>i</sub>) * log(p(X<sub>i</sub>)/q(X<sub>i</sub>))) = SUM(p(X<sub>i</sub>) * log(p(X<sub>i</sub>)) - SUM(p(X<sub>i</sub>) * log(q(X<sub>i</sub>)) = 样本信息熵(固定值) - SUM(p(X<sub>i</sub>) * log(q(X<sub>i</sub>)) </br>

交叉熵的值只与第二项 - SUM(p(X<sub>i</sub>) * log(q(X<sub>i</sub>))有关，带入p(X<sub>i</sub>)，q(X<sub>i</sub>)到该项，得到</br>
> -SUM(p(X<sub>i</sub>)\*log(q(X<sub>i</sub>)) = -SUM(Y<sub>i</sub> * log(1/(1 +e<sup>-1\*sigmod(θ<sup>T</sup>·X))</sup>)</br>
 
而这个值正是正例Y<sub>i</sub>=1、负例Y<sub>i</sub>=0的负对数似然函数，进而推导出：</br>
> 相对熵取最小值 <=> 负对数似然函数取最小值 <=> 对数似然函数取最大值 

