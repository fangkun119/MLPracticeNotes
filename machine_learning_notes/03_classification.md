
# 03. Classification

##1.用随机梯度下降分类器（SGDClassifier）训练模型
优势: 计算速度快<br/>
注意: 

* 数据要先shuffer成随机排列（因为数据集可能是按某种特定顺序预先排序的）
* 进行交叉验证，但对于分类任务，测试集分类准确率不足以评价模型的效果，必须与召回率一起看；或者用ROC曲线来评估

##2.准确率召回率
交叉验证输出混淆矩阵，格式为: 

> [[TN, FP]<br/>
> [FN, TP]]

对于预测结果，从预测是正例的这部分的角度来看: <br/>
> 预测精度，对于预测为正例的样本，有多少是正确预测的: `TP/(TP + FP)` <br/>
> 召回率，对于所有正例样本，有多少被预测出来了: `TP/(TP + FN)` <br/>

除了混淆矩阵，也可以使用框架的API直接得到准确率和召回率
 
##3.用PR（准召率）曲线评估模型性能
方法1:  F1分数、即准确率和召回率的**谐波**平均值 
> `F1 = 2 / (1/准确率 + 1/召回率)` </br>
> 用于在相近准确率、召回率的模型之间进行权衡 </br>
> 有时更关心准确率，或者更关心召回率，此时F1就不适用

方法2:  根据准召率曲线来决定<br/>
> * 选择阈值：<br/>
> 用`cross_val_predicct()`函数获得所有可能阈值下对应的准确率、召回率, 绘制准确率、召回率随阈值变化的情况、选择一个阈值<br/>
> 绘制准确率随召回率变化的情况、使用准确率即将陡峭下降位置对应的阈值, 获得预测的分数
> * 预测时不再调用`predict()`方法，而是调用`decision_function()`来得到预测分数

##4.ROC曲线：把预测结果为P的样本分成TP,FN两份分别加以研究
受试者工作特征曲线：真正类率率随假正类率变化的情况 <br/>
> 真正例率：正例样本中有多少被正确预测（找到了百分之多少的正例，等于召回率：`TP/(TP + FN)`）<br/>
> 假负例率：负例样本中有多少被错误预测（预测正例中混入了多少负例：`FN/(TN + FP)`) 

通过比较ROC曲线下面积来比较模型效果的好坏

##5.ROC曲线与PR(准召率)曲线
取决于关注哪种错误对模型效果照成的影响 <br/>
> * PR曲线：关注FP样本影响（精度低是FP导致的，随着召回率的上升，FP也会上升，如果上升过快，PR曲线就不好）<br/>
> * ROC曲线：关注FN样本照成的影响（ROC把预测为正例`P`的样本的分成`TP`、`FN`两份加以研究）

##6.用二元分类器做多分类任务
* OvA（one-versus-all）：一对多叠加，N个类别，需要训练N-1个二元分类器<br/>
* OvO（one-versus-one）：一对一组合，N个类别，需要训练N(N-1)/2个二元分类器<br/>

> sklearn检测到给二元分类器进行多分类任务时，会自开启OvA（SVM除外、会开启OvO）<br/>
> 也可用OneVsOneClassifier或者OneVsRestClassifier装饰器强制二元分类器开启多分类模式

##7.多分类器错误分析及解决
错误分析
> 交叉验证后输出混淆矩阵，对混淆矩阵进行可视化（将错误率转成颜色灰度），输出到图像中，查看哪两个类别容易分错（例如5和3总是容易分错）

解决
> * 收集更多样本，开发新特征（例如计算闭环数量），对样本图片进行预处理
> * 打印分错样本分析单个错误的原因（例如MINST手写数字数据集，发现3旋转后就与5的像素点非常像，因此添加预处理步骤，保证输入图片不会有倾斜）

##8.多标签分类
每个标签（Y向量）都有多个值，例如（数字是否大于7，数字是否是偶数）<br/>
Sklearn会自动识别Y向量是否有多个标签

##9.多输出分类
是多标签分类的泛化<br/>
> 例如，一个图片降噪应用，输入有噪点的图片，输出无噪点的图片。
输出是多个标签（一个像素点一个标签）、标签值是灰度值（0-255）

##10.特征筛选

特征筛选的方法有很多，主要包含在`ScikitLearn`的`feature_selection`库中

* 比较简单的有通过F检验（`f_regression`）来给出各个特征的F值和p，值，从而可以筛选变量（选择F值大的或者p值小的特征）
* 其次有递归特征消除（RecursiveFeatureElimination，RFE）
	
	> 反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），把选出来的特征放到一边，然后在剩余的特征上重复这个过程
	
* 稳定性选择（StabilitySelection）等比较新的方法

	> 主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断重复，最终汇总特征选择结果。比如，可以统计某个特征被认为是重要特征的频率


详细内容及代码：张良均,等. Python数据分析与挖掘实战 (大数据技术丛书) (Chinese Edition) (Kindle Locations 1626-1629). 机械工业出版社. Kindle Edition. 
