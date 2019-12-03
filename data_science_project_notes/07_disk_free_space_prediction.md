# 应用系统负载分析与磁盘容量预测

## 背景

分析存储设备中磁盘容量预测，可预测磁盘未来几天的磁盘占用量，根据需求设置不同预警等级，提供定制化预警提示

![11_disk_prediction_01_background.jpg](img/11_disk_prediction_01_background.jpg)

## 数据

字段说明

![11_disk_prediction_02_data_field.jpg](img/11_disk_prediction_02_data_field.jpg)

数据样例
![11_disk_prediction_03_data_sample.jpg](img/11_disk_prediction_03_data_sample.jpg)

完整数据: [data/discdata.xls](data/discdata.xls) 

## 过程

![11_disk_prediction_04_process.jpg](img/11_disk_prediction_04_process.jpg)

<b>数据抽取</b>：从数据源中选择性抽取历史数据与每天定时抽取数据。<br/>
<b>数据探索与预处理</b>：对抽取的数据进行周期性分析以及数据清洗、数据变换等操作后，形成建模数据。<br/>
<b>建模</b>：采用时间序列分析法对建模数据进行模型的构建，利用模型预测服务器磁盘已使用情况。<br/>
<b>应用</b>：应用模型预测磁盘使用情况，向管理员发送报警。<br/>

## 数据抽取

根据TARGET_ID（磁盘已使用大小、磁盘容量），ENTITY（C：，D：），COLLECTIME（采集时间）采集VALUE字段

## 数据探索分析

采用时序分析法建模，为了建模需要，要探索数据的平稳性（通过时序图可以初步发现数据的平稳性）。<br/>
针对服务器已使用大小，以天为单位进行周期性分析<br/>

![11_disk_prediction_05_time_series_diagram.jpg](img/11_disk_prediction_05_time_series_diagram.jpg)

## 数据预处理

1. 磁盘容量（固定值、不考虑磁盘扩容的情况下）去重：每个磁盘容量数据合并成1条<br/>
2. 主key合并：将<TARGET_ID, ENTITY, NAME>合并成一个字段

合并前、合并后的数据：
![11_disk_prediction_06_key_merge.jpg](img/11_disk_prediction_06_key_merge.jpg)

[代码：主key合并](code/11-1_attribute_transform.py)

3. 训练集、测试集划分（后5条为测试集）

## 模型构建过程

![11_disk_prediction_07_model_training.jpg](img/11_disk_prediction_07_model_training.jpg)

1. 数据平稳<br/>

	> 对观测值序列进行平稳性检验<br/>
	> 如果不平稳，则对其进行差分处理直到差分后的数据平稳<br/>

2. 白噪声检验及模式识别<br/>

	> 数据平稳后、进行白噪声检验<br/>
	> 如果没有通过白噪声检验，就进行模型识别，识别其模型属于AR、MA和ARMA中的哪一种模型<br/>
	
3. 参数估计，误差分析<br/>

	> 通过BIC信息准则对模型进行定阶，确定ARIMA模型的p，q参数<br/>
	> 在模型识别后需进行模型检验，检测模型残差序列是否为白噪声序列。<br/>
	> 如果模型没有通过检测，需要对其进行重新识别，对已通过检验的模型采用极大似然估计方法进行模型参数估计。

最后，应用模型进行预测，将实际值与预测值进行误差分析。如果误差比较小（误差阈值需通过业务分析进行设定），表明模型拟合效果较好，则模型可以结束。反之需要重新估计参数。

## 模型构建各环节用到方法

### 平稳性检验

<b>用途</b>：为了确定原始数据序列中没有随机趋势或确定趋势，需要对数据进行平稳性检验，否则将会产生“伪回归”的现象。<br/>
<b>方法</b>：采用单位根检验（ADF）的方法或者时序图的方法进行平稳性检验<br/>

[代码：平稳性检验](code/11-2_stationarity_test.py)

检验结果：<br/>
![11_disk_prediction_08_stationarity_test.jpg](img/11_disk_prediction_08_stationarity_test.jpg)

### 白噪声检验：

<b>用途</b>：为了验证序列中游泳的信息是否已经被提取完毕，需要对序列进行白噪声检验（如果序列检验为白噪声序列，就说明序列中有用的信息已经被提取完毕了，剩下的全是随机扰动，无法进行预测和使用）<br/>
<b>方法</b>：采用LB统计量的方法进行白噪声检验

[代码：白噪声检查](code/11-3_whitenoise_test.py)

检验结果：<br/>

![11_disk_prediction_09_white_noise_test.jpg](img/11_disk_prediction_09_white_noise_test.jpg)

### 模式识别

### 模型检验

### 模型预测

## 模型评价

## 模型应用










