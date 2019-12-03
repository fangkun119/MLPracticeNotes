# 应用系统负载分析与磁盘容量预测

## 背景

分析存储设备中磁盘容量预测，可预测磁盘未来占用量，根据需求设置不同预警等级，提供定制化预警提示

![11_disk_prediction_01_background.jpg](img/11_disk_prediction_01_background.jpg)

## 数据

字段说明

![11_disk_prediction_02_data_field.jpg](img/11_disk_prediction_02_data_field.jpg)

数据样例
![11_disk_prediction_03_data_sample.jpg](img/11_disk_prediction_03_data_sample.jpg)

完整数据: [data/discdata.xls](data/discdata.xls) 

## 过程

![11_disk_prediction_04_process.jpg](img/11_disk_prediction_04_process.jpg)

数据抽取：从数据源中选择性抽取历史数据与每天定时抽取数据。<br/>
数据探索与预处理：对抽取的数据进行周期性分析以及数据清洗、数据变换等操作后，形成建模数据。<br/>
建模：采用时间序列分析法对建模数据进行模型的构建，利用模型预测服务器磁盘已使用情况。<br/>
应用：应用模型预测磁盘使用情况，向管理员发送报警。<br/>




