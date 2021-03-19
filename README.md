<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [1. NLP](#1-nlp)
- [2. 机器学习模型训练](#2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
- [3. 机器学习原理](#3-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8E%9F%E7%90%86)
- [4. 深度学习模型训练](#4-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
- [5. 小案例](#5-%E5%B0%8F%E6%A1%88%E4%BE%8B)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 1. NLP

> [https://github.com/fangkun119/nlpia](https://github.com/fangkun119/nlpia)
>
> 1. [Overview](https://github.com/fangkun119/nlpia/blob/master/notes/part_00_resource.md)
>
>     NLP Application使用领域，与正则表达式和规则系统对比，语法-词袋表征的局限，聊天机器人Pipe Line，现有一些NLP系统评估（IO深度、IO广度）
>
> 2. [分词](https://github.com/fangkun119/nlpia/blob/master/notes/part_01_wordy_machines_02_extract_word.md)
>
>     (1) 缩写、词干、前缀后缀、复数、不可见的词、短语识别、词频、EOS符号、数字、标点符号、停用词、2-gram、3-gram、大小写处理、词干提取、词型还原等技巧
>
>     (2) One-Hot编码，词袋模型，词向量距离
>
>     (3) 基于规则的情绪识别，基于朴素贝叶斯的情绪识别（？？？）
>
> 3. [TF-IDF词向量](https://github.com/fangkun119/nlpia/blob/master/notes/part_01_wordy_machines_03_doc_vector.md)
>
>     词袋（BOW），TF-IDF， Zipf定律，TF-IDF向量相似度计算（余弦距离、BM25）、文档向量
>
> 4. [语义分析](https://github.com/fangkun119/nlpia/blob/master/notes/part_01_wordy_machines_04_sementic_analysis.md)
>
>     (1) 主题向量的用途：用于分类（本章demo）、相似度度量、语义聚类
>
>     (2) 线性判别分析（`LDA：Linear Discriminant Analysis`）
>
>     (3) 潜语义分析（`LSA：Latent semantic analysis`）：理解`LSA`/`SVD`/`PCA`，使用recreate-accuration确定保留的主题数，LSA使用的数据量，使用LSA生成短信的主题向
>
>     (4) 隐狄利克雷分布（`LDiA: Latent Dirichlet allocation`）：LDA和LDiA之间的选择；LDiA的原理和超参数（文档平均长度、主题数）；使用LDiA生成短信的主题向量；LDiA共线性报警（collinear Warning）问题
>
>     (5) 对比使用`LDA(LDiA(TF-IDF))`、`LDA(LSA(TF-IDF))`和使用`LDA(TF-IDF)`对垃圾短信分类
>
>     (6) 距离函数和相似度度量
>
>     (7) 学术界的研究和改进方向：主题向量在语义搜索中的使用，主题向量的改进
>
> 5. [神经网络](https://github.com/fangkun119/nlpia/blob/master/notes/part_02_deep_learning_05_neural_networks.md)
>
>     (1) 感知机训练原理及反向传播
>
>     (2) Error Surface、批量学习、随机梯度下降、Mini Batch梯度下降
>
>     (3) 使用Keras编写神经网络代码、输入归一化
>
> 6. [Word2Vec](https://github.com/fangkun119/nlpia/blob/master/notes/part_02_deep_learning_06_word2vec.md)
>
>     (1) 使用word2vec进行语义查询和语义推理，对比word2vec和LSA
>
>     (2) word2vec的两种训练方法：Skip-Gram和CBOW、频繁二元组降采样、Negative Sampling
>
>     (3) 使用预训练的word2vec模型
>
>     (4) 训练自己的domain-specified word2vec模型
>
>     (5) 对比word2vec、Glove、FastText
>
>     (6) 使用word2vec查询词向量、降维到2维来可视化单词之间的关系
>
>     (8) 使用doc2vec计算文档相似度
>
> 7. [CNN用于NLP](https://github.com/fangkun119/nlpia/blob/master/notes/part_02_deep_learning_07_cnn.md)
>
>     (1) CNN所解决的问题，理解CNN网络结构，训练原理
>
>     (2) CNN例子：影评分类（Keras）
>
> 8. [RNN](https://github.com/fangkun119/nlpia/blob/master/notes/part_02_deep_learning_08_rnn.md)
>
>     (1) RNN所解决的问题，理解RNN网络结构及时间片展开
>
>     (2) 只关心样本最后一个token的预测输出、和关心每个token的预测输出两种应用方式
>
>     (3) RNN例子：影评分类（Keras）
>
> 9. [LSTM用于NLP](https://github.com/fangkun119/nlpia/blob/master/notes/part_02_deep_learning_09_LSTM.md)
>
>     (1) LSTM按时间片展开、以及LSTM Cell的结构、梯度消失和梯度爆炸问题
>
>     (2) LSTM例子：影评分类（Keras）代码及未知词处理方法
>
>     (3) 使用LSTM生成风格相似的文本及例子：生成莎士比亚风格的诗词
>
>     (4) 其他记忆网络：GRU、窥视孔连接
>
>     (5) 多层LSTM（Stacked LSTM）
>
> 10. Seq2Seq Model以及Attention (doing)
>
> 11. 信息提取、Named Entity Extraction以及问答系统（todo）
>
> 12. 对话引擎（todo）
>
> 13. 可扩展性（优化、并行、批处理）（todo）
>
> 14. 其他：[文档](https://github.com/fangkun119/nlpia/blob/master/notes/part_00_nlpia_docs.md)，[资源](https://github.com/fangkun119/nlpia/blob/master/notes/part_00_resource.md)

## 2. 机器学习模型训练

> 1. [机器学习介绍](machine_learning_notes/01_introduction.md) 
> 2. [机器学习项目](machine_learning_notes/02_machine_learning_project.md) 
> 3. [分类](machine_learning_notes/03_classification.md) 
> 4. [线性回归、Logistics回归、SoftMax以及模型训练技巧](machine_learning_notes/04_model_training_and_linear_regression.md) 
> 5. [SVM](machine_learning_notes/05_svm.md) 
> 6. [决策树](machine_learning_notes/06_decision_tree.md) 
> 7. [集成学习：Bagging以及随机森林](machine_learning_notes/07_ensembled_learning_bagging_random_forest.md) 
> 8. [集成学习：Boosting](machine_learning_notes/08_ensembled_learning_boosting.md) 
> 9. 时间序列: [常用算法](https://www.kaggle.com/fangkun119/learn-time-series-analysis-in-python) 、[Facebook Prophet](https://www.kaggle.com/fangkun119/topic-9-part-2-time-series-with-facebook-prophet)

## 3. 机器学习原理

> |              内容               |                             摘要                             |                     网页笔记（MarkDown）                     |                     PDF（手写笔记扫描）                      |
> | :-----------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
> |            线性代数             |                                                              |                                                              | [线性代数](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Linear_Algebra.pdf) |
> |        数学分析及概率论         |           [熵](theory_note/Appendix_01_entropy.md)           |   [概率论](theory_note/Appendix_10_probability_theory.md)    | [数学分析及概率论1](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Mathematics_Analysis_and_Probability_Theory_1.pdf)    [概率论2](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Probability_Theory_2.pdf) |
> | 线性回归、Logistic回归、SoftMax | [分类模型](theory_note/Appendix_02_classification_algorithms.md)  [线性回归](theory_note/Appendix_03_linear_regression.md) | [线性回归](theory_note/Appendix_08_linear_regression.md) [Logistic回归与SoftMax](theory_note/Appendix_09_logistic_regression.md) | [PDF 1](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Linear_Regression_Logistic_Regression_and_SoftMax_1.pdf)  [PDF 2](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Linear_Regression_Logistic_Regression_and_SoftMax_2.pdf) |
> |        决策树和随机森林         |                                                              | [决策树和随机森林](theory_note/Appendix_11_decision_tree_and_random_forest.md) | [决策树和随机森林](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Tree_and_Forest.pdf) |
> |            Boosting             |                                                              | [XGBoost](theory_note/Appendix_05_xgboost.md) [AdaBoost](theory_note/Appendix_06_adaboost.md) | [Adaboost and GBDT](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Boosting_AdaBoosting_GBDT.pdf) |
> |               SVM               |                                                              |            [SVM](theory_note/Appendix_04_svm.md)             | [SVM](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/SVM.pdf) |
> |              聚类               |                                                              |        [聚类](theory_note/Appendix_07_clustering.md)         | [聚类](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Clustering_Algorithms.pdf) |
> |               EM                |                                                              |                                                              | [EM](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/EM.pdf) |
> |            主题模型             |                                                              |                                                              | [主题模型](https://github.com/fangkun119/MLTheoryNotes/blob/master/notes/Topic_Model.pdf) |

## 4. 深度学习模型训练

> 1. [Tensor Flow](deep_learning_notes/10_hands_on_tensorflow.md) 
> 2. [ANN](deep_learning_notes/11_ann.md) 
> 3. [DNN模型训练技巧](deep_learning_notes/12_dnn_train_skills.md) 
> 4. [分布式Tensor Flow](deep_learning_notes/13_distributed_tensorflow.md) 
> 5. [CNN](deep_learning_notes/14_cnn.md) 
> 6. [RNN](deep_learning_notes/15_rnn.md) 
> 7. [自编码机](deep_learning_notes/16_auto_encoder.md) 
> 8. [强化学习](deep_learning_notes/17_reinforcement_learning.md) 

## 5. 小案例

> 1. [数据科学项目方法概述](./data_science_project_notes/01_data_science_project.md)
> 2. [电力窃取用户识别](./data_science_project_notes/02_electric_power_stealing_user_identification.md)
> 3. [航空客户价值分析](./data_science_project_notes/03_airline_customer_value_analysis.md)
> 4. [中医证型关联规则挖掘](./data_science_project_notes/04_TCM_syndromes_association_rule_mining.md)
> 5. [家用热水器用户行为识别](./data_science_project_notes/06_household_appliances_user_behavior_analysis.md)
> 6. [系统磁盘可用容量预测](./data_science_project_notes/07_disk_free_space_prediction.md)
> 7. [电子商务网站用户行为分析及服务推荐](./data_science_project_notes/08_e_com_recommendation.md)

