# 分类模型算法推导的简要概述

## 1.Logistic回归
(1) <b>假设函数(hypothesis function)</b>：<br/>
把θ<sup>T</sup>.X作为sigmoid函数的参数，从而将值域从`(-∞,∞)`$映射到`(0,1)`，其中0映射到0.5 <br/>
(2) <b>似然函数(likelyhood function)</b>: <br/>
映射后H(θ, X) = sigmoid(θ<sup>T</sup>·X) 是样本为正例的预测概率, 根据样本标签Y<sub>i</sub>取值有1，0两种<br/>
> Y<sub>i</sub>=0 时使用 1 - H(θ, X) = 1 - sigmoid(θ<sup>T</sup>·X) 表示样本为负例的预测概率<br/>
> Y<sub>i</sub>=1 时使用 sigmoid(θ<sup>T</sup>·X) 表示样本为正例的预测概率

合并两个公式得到 P(Y<sub>i</sub>|X<sub>i</sub>, θ) = (H(θ,X<sub>i</sub>)^Y<sub>i</sub>) * (1 - H(θ,X<sub>i</sub>)^(1-Y<sub>i</sub>))，值取最大时预测最准确<br/>
带入所有样本，就可以得到似然函数L(θ)（样本和标签已知的条件下，表示模型参数θ取特定值的概率的函数）
> 公式：L(θ)= ∏(H(θ,X<sub>i</sub>)<sup>Y<sub>i</sub></sup>)*(1 - H(θ,X<sub>i</sub>))<sup>(1-Y<sub>i</sub>)</sup>)

(4) 取对数得到对数似然函数<br/> 
> log(L(θ))=∑(Y<sub>i</sub>\*log(H(θ,X<sub>i</sub>))+(1-Y<sub>i</sub>)\*log(1 - H(θ,X<sub>i</sub>)))

(5) 似然函数取最大值时，模型的精确度最高，进而可以用求偏导、梯度上升的方法求解 <br/>
备注1：H(θ,X)表示样本为正的预测概率, 是来自作者的设计思路。
> 他想设计一个函数使得log (P<sub>positive</sub>/(1-P<sub>positive</sub>))是线性函数，进而能够用θ<sup>T</sup>·X表示，最终他使用了sigmoid函数。<br/>
> 推导过程: log (P/(1-P)) = log (H(θ,X)/(1 - H(θ,X)) = log (1 / e<sup>(-θ<sup>T</sup>·X)</sup>)</sup> = θ<sup>T</sup>·X

备注2：Logistic Regression 的损失函数，就是负的对数似然函数
> 对数似然函数（梯度上升）取到最大值 <=> 损失函数 (梯度下降）取到最小值

## 2.Softmax回归
K分类、第K类参数为为列向量Theta_k，所有类别的参数列向量可以组成矩阵Theta_k_n
(1) 	类别取K的概率密度: P_k = P(c=k | X; Theta) = e^(Transpose(Theta_k).X) / SUM(e^(Transpose(Theta_i).X)) 
(2) 	然而softmax并没有使用 P_k / SUM(P_i) 来建模，而是用 e^P_k / SUM(e ^ P_i) 来近似， 因为P_i < P_j时，e^P_i < e^P_j,  e^P_i / SUM(e^P_i + e^P_j) < e^P_j / SUM(e^P_i + e^P_j)，不影响效果；且方便后续的推导
SUM(e ^ P_i))^Y_k_i 可以表示对于类别为k的样本，预测正确的概率
(3)	假设样本间相互独立，得到似然函数
LikelyHood(Theta) = Product_i(Product_k ( P(c=k | X_i, Theta)^Y_i_k) )) = Product_i(Product_k ( (e^P_k / SUM(e ^ P_i))^Y_k_i ))
(4)	对数似然
Log(LikelyHood(Theta)) = SUM_i (SUM_j ( Y_k_i * (P_k - log(SUM(e ^ P_i))) )) 
(5)	对这个对数似然函数求偏导，做梯度上升，得到其取最大值时的参数Theta
备注1:  k = 2 时，令Theta = Theta_1 - Theta_2， Softmax回归退化为logistic回归
P_1 = e^Tranpose(Theta_1).X / (e^Tranpose(Theta_1).X  + e^Tranpose(Theta_2).X ) 
      = 1 / (1 + e^Tranpose(- (Theta_1 - Theta_2)).X )  
      = 1 / (1 + e^Tranpose(-Theta) . X)

#3.Logistic/Softmax与相对熵
(1) 相对熵
相对熵（也叫交叉熵、鉴别信息、Kullback熵，Kullback-Leible散度，K-L距离）
p(x)是我们在研究的概率分布（例如样本概率），q(x)为一个基准分布（例如似然概率）
p(x)与q(X)的相对熵D(p||q)，可以理解为对log(P(x)/Q(x))求数学期望，即
D(p||q) = SUM(p(x)*log(p(x)/q(x))) 
(2) 在logistic regression中
p(x)被建模为样本类别的概率分布；q(x)被建模为似然概率，即：
	p(X_i) = Y_i
	q(X_i) = sigmod(transpose(Theta).X) = 1 / (1 + e ^ (-1 * sigmod(transpose(Theta).X))) 
D(p||q) = SUM(p(X_i)*log(p(X_i)/q(X_i))) = SUM(p(X_i)*log(p(X_i)) - SUM(p(X_i)*log(q(X_i)) = 样本信息熵(定值) - SUM(p(X_i)*log(q(X_i))
交叉熵的值只与第二项-SUM(p(x)*log(q(x)))有关，带入p(X_i)，q(X_i) 得到
	- SUM(p(X_i)*log(q(X_i)) = - SUM(Y_i * log (1 / (1 + e ^ (-1 * sigmod(transpose(Theta).X)))) 
	而这个值正是正例Y_i=1、负例Y_i=0的负对数似然函数
推导出：
	相对熵取最小值 <=> 负对数似然函数取最小值 <=> 对数似然函数取最大值 
(3) 相对熵用p(x)/q(x)还是q(x)/p(x)
log(p(x)/q(x))取值区间段，希望大部分落在中间的B段（1 <= p(x)/q(x) < some_value)
根据第二项的公式SUM(p(x)*log(q(x)))，对于p(x)=0时不影响取值，p(x)=1时如果q(x)接近0会得到一个非常大的惩罚值