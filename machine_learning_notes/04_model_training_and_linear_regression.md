# CH04 模型训练过程（线性回归为例）

## 1.线性回归：
**线性回归求解方法：** <br/>
方法1: 使用“闭式”方程直接求解模型参数、计算量大
> θ=(X<sup>T</sup>·X)<sup>-1</sup>·X<sup>T</sup>·y 

方法2. 使用梯度下降（GD）
> 似然函数: Y=θ<sup>T</sup>·X+b <br/>
> MSE损失函数（m是样本数）：1/m * SUM<sub>i</sub>(θ<sup>T</sup>·X<sub>i</sub>-Y<sub>i</sub>)<sup>2</sup>， 

**线性回归原理：**<br/>
prediction = L(θ<sup>T</sup>·X) + error<br/>
> L表示似然函数；error表示误差；预测值可以用似然函数和误差来表示

假定总误差值服从相同的高斯分布，以0为中心，方差是r，单个样本预估误差表示为：
> error<sub>i</sub> = y<sub>i</sub> - prediction<sub>i</sub><br/>

因此(Y<sub>i</sub> - Predict(X<sub>i</sub>))=(Y<sub>i</sub>-θ<sup>T</sup>·X<sub>i</sub>)也可以写成高斯分布的形式，以0为中心，方差也是r。因为假定样本相互独立，则误差值也相互独立，同时他们服从相同分布，根据大数定理：所有样本的总误差也服从中心点为0、方差为r的高斯分布，该分布值可用样本分布值的乘积来表示。如果让这个分布的概率密度值取最大，那么就位于高斯分布的中心，也就是误差最小的地方。为了方便求解，对上面的分布取对数（以便将乘法转换为加法）。消去常数项和固定系数，剩余的变量部分就是 -SUM(Y<sub>i</sub>-θ<sup>T·X<sub>i</sub></sup>)<sup>2</sup>，让这个值取最大值，等价于让（去掉头部的符号后的）SUM(Y<sub>i</sub>-θ<sup>T· X<sub>i</sub></sup>)<sup>2</sup>取最小值<br/>，进而也就可以找到到让误差位于中心点(概率密度最大位置)的系数θ。<br/>
**正则项：**<br/>
为了抑制参数θ的取值，可以添加正则项到上面的函数中，其中Lasso具有特征选择能力，ƛ是控制正则化项权重的超参数<br/>
> **Lasso**：让 SUM(Y<sub>i</sub>-θ<sup>T</sup>·X<sub>i</sub>)<sup>2</sup>+ƛ* SUM(|θ<sub>j</sub>|)取最小值<br/>
> **Ridge**：让 SUM(Y<sub>i</sub>-θ<sup>T</sup>·X<sub>i</sub>)<sup>2</sup>+ƛ* SUM(θ<sub>j</sub><sup>2</sup>)取最小值<br/>
> **ElisticNet**: 让 SUM(Y<sub>i</sub>-(θ<sup>T</sup>·X<sub>i</sub>)<sup>2</sup>+ƛ\*(p\*SUM(|θ<sub>j</sub>|)+(1-p)\*SUM(θ<sub>j</sub><sup>2</sup>) 取最小值

**线性回归效果评估判定系数**<br/>

* **均方误差(MSE=RSS, 即残差平方和)**：SUM((Predict(ƛ,X<sub>i</sub>)-Y<sub>i</sub>)<sup>2</sup>)
* **R<sup>2</sup>判定系数**：R<sup>2</sup>=1-RSS/TSS=1-(SUM((Predict(ƛ,X<sub>i</sub>)-Y<sub>i</sub>)<sup>2</sup>))/(SUM(Y<sub>i</sub>-Y<sub>avg</sub>)<sup>2</sup>) 

	> 残差平方和高可能有两个原因：(a) 模型误差大 (b) 样本本身发散Y值震荡大<br/>
	> RSS：预测误差平方和<br/>
	> TSS：(样本均值，样本值)平方和<br/>
	> R<sup>2</sup>值越大说明模型越准确，越小说明误差越大，一般取值在[0,1]，为负值说明误差极大

* **ESS判定系数（又叫回归平方和）**：ESS = SUM((Y<sub>i,predict</sub>-Y<sub>avg</sub>)<sup>2</sup>) 
	> TSS(样本伪方差) >= ESS(预测值) + RSS(残差平方和) <br/>
	> 仅当无偏估计时才成立
	
* **MAE**: 平均绝对误差（MeanAbsoluteError，MAE）
* **RMSE**: 均方根误差
* **平均绝对百分误差**（MeanAbsolutePercentageError，MAPE）

## 2.梯度下降
超参数（学习率）：太低收敛慢、太高可能会导致损失值发散越来越大</br>
梯度下降一定能收敛的前提：成本函数是一个凸函数、这样的函数不存在局部最小，只有一个全局最小值</br>
影响梯度下降收敛速度的另一个前提：各个特征尺寸差别不大（圆形的碗而不是细长的碗）

## 3.批量梯度下降
用整个训练集所有的m个样本、计算在每个Thetha_i方向上的偏导数，得到梯度向量
用梯度向量乘以学习率步长来更新Theta向量
为了找到合适的学习率，可以使用网格搜索，但是要限制迭代次数（可以开始时大一些，当Theta范数的变化小于一定值时中断算法）
缺点是训练计算量庞大

## 4.随机梯度下降
每一步不再考察全部m个样本、而是从样本中随机选择一个、计算和更新梯度<br/>

* 计算速度比批量梯度下降快 
* 但是成本函数下降将不再平稳，而是不断上上下下但是整体上还是在慢慢下降
* 另一个特点是有助于跳出局部最优值，但定位不出最小值，解决办法是采用模拟退火（开始下降快，之后下降慢）

## 5.小批量梯度下降（Mini-Batch)
介于两者之间

## 6.多项式回归
用线性回归来拟合非线性数据

## 7.判断模型是欠拟合还是过拟合
方法1：交叉验证、比较训练集和测试集的模型效果<br/>
方法2：观察学习曲线、即训练集、测试集Loss值随训练集变化的情况

## 8.过拟合问题解决
* 简化模型
* 更多训练数据
* 减少数据的噪音
* 正则化超参数
* 早期停止法
* 降维
* 基于业务逻辑来杂凑组合特征

## 9.线性模型的正则化参数
正则化：通过约束模型特征的权重来防止过拟合

### (1) Ridge：ɑ * SUM<sub>i</sub>(θ<sub>i</sub><sup>2</sup>)
训练时添加到损失函数中，即MSE(θ)+ɑ* SUM(θ<sub>i</sub><sup>2</sup>), 超参数ɑ控制正则化的程度。注意：

* 预测时不能将正则化参数添加在似然函数中
* 必须对数据进行缩放（例如使用StandardScaler）因为对输入特征的大小敏感

求解带Ridge正则项的线性回归，与普通线性模型一样，可以用闭式方程求解，也可以用梯度下降求解

### (2) Lasso回归：ɑ * SUM<sub>i</sub>(|θ<sub>i</sub>|)
训练时添加到损失函数中，MSE(θ)+ɑ * SUM<sub>i</sub>(|θ<sub>i</sub>|), 超参数ɑ控制正则化的程度，注意：

* Lasso倾向于完全消除掉最不重要特征的权重（置零），换句话说会自动执行特征选择并输出一个稀疏模型
* Lasso在训练后期会发生斜率突变（P124，也是特征选择的效果），损失值会发生振动，需要逐渐降低学习率来保证全局最小收敛

### (3) Elastic Net：介于Ridge和Lasso之间，
用Ridge和Lasso正则项加权求和来作为Elastic Net的正则项，其中超参数r就是这个求和权重，用来控制是更接近Ridge还是更接近Lasso

### (4) 如何选择正则项
* 大多数情况下应该使用正则项（避免使用纯线性回归）
* Ridge是不错的默认选择
* 如果觉得实际会用到的特征只有少数几个，应该更倾向于Lasso或者Elastic Net
* Elastict一般优于Lasso，特别当特征数量超过训练实例数量或者几个特征强相关时Lasso可能表现得非常不稳定

## 9.线性回归的早期停止法
**(1) 对于批量梯度下降**</br>
增加轮次不断根据训练集全量数据计算梯度更新θ，训练集Loss会一直变小，测试集Loss变小到某个时刻后会开始增大（发生过拟合），在测试集Loss即将开始变大的时候，立刻停止训练</br>
**(2) 对随机梯度下降、小批量梯度下降**</br>
测试集Loss下降不会那么平滑（会抖动）、很难知道是否已经到了最小值，
一种解决办法是等待验证误差超过最小值一段时间之后再停止

## 10. Logistic回归
概率估算： 
> P = sigmod(θ<sup>T</sup>·X)

sigmoid函数：
> sigmod(θ<sup>T</sup>·X) = 1/(1 + exp(-θ<sup>T</sup>·X))

模型预测： 
> y = 0 当 P < 0.5  
> y = 1 当 P >= 0.5 

单个样本的损失函数：借助log函数的特点、当分错时loss值会非常大
> -log(P) 当y-1时；
> -log(1-P)当y等于0时 

整个模型的损失函数：
> -(1/m)* SUM(Y<sub>i</sub> * log(P<sub>i</sub>)+(1-y)log(1-P<sub>i</sub>))<br/>

模型损失函数的偏导数：
> (1/m)*SUM(logistic(θ<sup>T</sup>·X<sub>i</sub>) - Y<sub>i</sub>).X<sub>i</sub>

Logistic回归也可以加正则化参数

## 11. Logistic回归决策边界可视化 P130
决策边界处有一段模糊地带，会有分错的样本

## 12.Softmax分类
样本X在类别k上的得分：
> s<sub>k</sub>(X) = θ<sub>k</sub><sup>T</sup>·Xθ

样本X属于类别k的概率：
> P<sub>x,k</sub> = sigmod( exp(θ<sub>k</sub><sup>T</sup>·X) / SUM (exp(θ<sub>i</sub><sup>T</sup>·X)))

样本预测：
> 对于样本x、类别k（属于[1,K]），看哪个k的P<sub>x,k</sub>值最大

单个样本的损失函数（交叉熵）：当样本x是类别k时，标签Y<sub>x,k</sub>=1，如果预测的P<sub>x,k</sub>很低接近0，log(P<sub>x,k</sub>)就会很大，得到一个很大的惩罚项
> -SUM<sub>k∈[1,K]</sub>(Y<sub>x,k</sub> * log (P<sub>x,k</sub>))  

整个模型的损失函数（交叉熵）： 
> -(1/m) * SUM<sub>i∈[1,m]</sub>(SUM<sub>k∈[1,k]</sub>(Y<sub>i,k</sub> * log (P<sub>i,k</sub>)))

损失函数的梯度向量： 
> (1/m) SUM<sub>i∈[1,m]</sub>((P<sub>i,k</sub>-Y<sub>i,k</sub>)·X<sub>i</sub>)

Softmax回归也可以加正则化参数，默认是L2参数（Ridge）


