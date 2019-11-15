# 线性回归算法推导简要介绍

## 1.线性回归原理：
prediction = L(θ<sup>T</sup>·X) + error<br/>
> L表示似然函数；error表示误差；预测值可以用似然函数和误差来表示

假定总误差值服从相同的高斯分布，以0为中心，方差是r，单个样本预估误差表示为：
> error<sub>i</sub> = y<sub>i</sub> - prediction<sub>i</sub><br/>

因此(Y<sub>i</sub> - Predict(X<sub>i</sub>))=(Y<sub>i</sub>-θ<sup>T</sup>·X<sub>i</sub>)也可以写成高斯分布的形式，以0为中心，方差也是r。因为假定样本相互独立，则误差值也相互独立，同时他们服从相同分布，根据大数定理：所有样本的总误差也服从中心点为0、方差为r的高斯分布，该分布值可用样本分布值的乘积来表示。如果让这个分布的概率密度值取最大，那么就位于高斯分布的中心，也就是误差最小的地方。为了方便求解，对上面的分布取对数（以便将乘法转换为加法）。消去常数项和固定系数，剩余的变量部分就是 -SUM(Y<sub>i</sub>-θ<sup>T·X<sub>i</sub></sup>)<sup>2</sup>，让这个值取最大值，等价于让（去掉头部的符号后的）SUM(Y<sub>i</sub>-θ<sup>T· X<sub>i</sub></sup>)<sup>2</sup>取最小值<br/>，进而也就可以找到到让误差位于中心点(概率密度最大位置)的系数θ。<br/>

## 2. 正则项
为了抑制参数θ的取值，可以添加正则项到上面的函数中，其中Lasso具有特征选择能力，ƛ是控制正则化项权重的超参数<br/>
> **Lasso**：让 SUM(Y<sub>i</sub>-θ<sup>T</sup>·X<sub>i</sub>)<sup>2</sup>+ƛ* SUM(|θ<sub>j</sub>|)取最小值<br/>
> **Ridge**：让 SUM(Y<sub>i</sub>-θ<sup>T</sup>·X<sub>i</sub>)<sup>2</sup>+ƛ* SUM(θ<sub>j</sub><sup>2</sup>)取最小值<br/>
> **ElisticNet**: 让 SUM(Y<sub>i</sub>-(θ<sup>T</sup>·X<sub>i</sub>)<sup>2</sup>+ƛ\*(p\*SUM(|θ<sub>j</sub>|)+(1-p)\*SUM(θ<sub>j</sub><sup>2</sup>) 取最小值

## 3. 线性回归效果评估判定系数

* **均方误差(MSE=RSS, 即残差平方和)**：SUM((Predict(ƛ,X<sub>i</sub>)-Y<sub>i</sub>)<sup>2</sup>)
* **R<sup>2</sup>判定系数**：R<sup>2</sup>=1-RSS/TSS=1-(SUM((Predict(ƛ,X<sub>i</sub>)-Y<sub>i</sub>)<sup>2</sup>))/(SUM(Y<sub>i</sub>-Y<sub>avg</sub>)<sup>2</sup>) 

	> 残差平方和高可能有两个原因：(a) 模型误差大 (b) 样本本身发散Y值震荡大<br/>
	> RSS：预测误差平方和<br/>
	> TSS：(样本均值，样本值)平方和<br/>
	> R<sup>2</sup>值越大说明模型越准确，越小说明误差越大，一般取值在[0,1]，为负值说明误差极大

* **ESS判定系数（又叫回归平方和）**：ESS = SUM((Y<sub>i,predict</sub>-Y<sub>avg</sub>)<sup>2</sup>) 
	> TSS(样本伪方差) >= ESS(预测值) + RSS(残差平方和) <br/>
	> 仅当无偏估计时才成立

