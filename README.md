# A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models
_This is the assignment of Data Mining course at my university. All the code for experiment is in ML-based_DM folder and all of detail information about problem, methods, experiment and results can be visited in my [Report.pdf](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/blob/main/Report.pdf) file._

## Quick Introduction and Experiment
Data Matching is the problem of finding structure data items that describe the same real-world enity, which has many applications such as identity management, risk management,
fraud detection and prevention, credit scoring/risk assessment and compliance monitoring.

In this assignment, we will **observe and compare the performance of data matching model based on traditional machine learning methods and deep learning approaches on three different type of datasets, namely clean structural, dirty and textual**.

## What we observed
* For _**clean structural datasets**_, we found that traditional machine learning-based baseline
models (Support Vector Machines (SVM), XGBoost, and Random Forest) were sufficient and exhibited comparable performance to deep learning models.

* For _**textual and dirty datasets**_, deep learning-based methods demonstrated
superior performance. This can be attributed to their ability to integrate natural language
processing capabilities from attention-based or recurrent neural network models.

## Experiment Result
### Dataset Overview
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/14dea8bd-1317-4721-abdb-2d78f54ad76e)

### Traditional Machine Learning Methods and Deep Learning Methods
**Traditional Machine Learning Methods**
* Naive Bayes
* Random Forest
* Decision Tree
* Support Vector Machine
* Logistic Regression
* XGBoost

**Deep Learning Methods**
* SIF
* RNN-based
* Attention-based
* Hybrid

_Detail of those methods' description can be found on my team report_

###  Dataset 1: iTunes-Amazon-1 (Structured)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/c0b6bcfe-d80b-4488-b5a9-2a56b0af7ca7)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/2f741607-b520-4a93-bdb0-eaebc3964ae4)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/a0e629a8-0006-4bb1-84c4-8314b7cb13f4)

### Dataset 2: DBLP-Scholar-1 (Structured)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/80ec0ac2-1b03-49c6-9577-5c82cd3836d9)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/82b29c67-e1ef-4342-b274-1146af32d689)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/2ee64b1f-b5f7-4bca-83c2-62eb65e94ce4)

### Dataset 3:  Abt-Buy (Textual)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/99c3a493-33c2-49f0-826a-c494ea754827)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/3ad74959-e16b-4bae-90f7-2a93348fc846)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/02b0b13e-80b0-4a5f-be4a-cabaaac697bd)

### Dataset 4: iTunes-Amazon-2 (Dirty)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/447fb3d5-5466-434b-b7c9-e7fea09310a5)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/9cc7bbea-3372-47ea-a28d-cfff2436ac9c)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/8712c38a-0e52-4830-be69-fcacd469a302)

###  Dataset 5: DBLP-Scholar-2 (Dirty)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/e022c50a-3363-454d-9ed1-d313c2946af7)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/780c2a43-b416-4f1b-a0f6-6e09e08ede2e)
![image](https://github.com/BinhTran-HCMUT/A-Comparative-Survey-on-Machine-Learning-based-Data-Matching-Models/assets/98327248/25865ff0-bcf5-479b-a5d4-9226bc04d133)

## Conclusion 
Based on our findings, we recommend that deep learning-based methods be applied when
dealing with textual and dirty datasets, as they offer improved performance. However, for
clean structural datasets, traditional machine learning models can be a viable option, providing
comparable results with lower time costs.
 
