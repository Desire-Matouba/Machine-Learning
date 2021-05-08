## MACHINE LEARNING FOR MEDICAL DIAGNOSIS

There are 2 types of cancer, one is Benign and other is Malignent. Malignent is most dangerous one. Early diagnosis of this cancer significantly increases the chances of survival. Machine learning models give more accuracy than doctors in diagnosis of cancer. So we are classifying the 2 types of cancer.

1)Data from Kaggle website is used. The dataset consists of variables that has the values of features of cancer images.

2)The dataset is divinded into test and train(80-20)

3)Three machine learning models are used to train the test data. SVM, Logistic Regression and Decision Tree.

4)Two evaluation methods are used, Confusion matrix and ROC-AUC curve to evaluate the model The code for each step is explained below.

## Steps to run the code 

Order of the code :

Data_Exploration
Support_vector_machine
Improving SVM Model
Logistic_Regression
Logistic Regression improvement
DecisionTree
Improving Decision tree model
ROC curve of best 3 models

## Explaining each code

Data_Exploration
Download the dataset from the link :https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

Import the file into juyter

Check the null values, if any value is not numeric then make it as zero-numeric

Changing the diagnosis column to 0 and 1 instead of M and B (Malignant and Benign) for future use. Delete the string column and keep the numeric column

Split for 0 and 1 is shown

All the 30 features scatter plot is shown

Correlation is shown for all the 30 features

Support_vector_machine
Here first, the dataset is split into Test and Train data (80-20%)

Further, from the Trian and Test data, diagnosis column is seperated.

There will be 4 datatsets now. X_train, y_train, X_test,y_test. Dataset with X will have all features and dataset y will have diagnosis column only.

SVC is used to fit the model for X_train and y_train.

Then prediction is done for X_test using svc_predict.

Theb the results of predicted value and y_test are compared.

Confusion matrix is plotted using the fuction

Accuracy, Precision and Recall of model is shown using the in-built function

Improving SVM Model
The data is standadised to [0,1] range and again trained anf tested the model.

Next, c and gamma values are changed for the model. This is done by using a funtion to get the optimised value.

Again the model is trained and tested. Confusion matrix is run for both the improvement methods

Logistic_Regression
Here first, the dataset is split into Test and Train data (80-20%)

Further, from the Trian and Test data, diagnosis column is seperated.

There will be 4 datatsets now. X_train, y_train, X_test,y_test. Dataset with X will have all features and dataset y will have diagnosis column only.

Logistic regression function is used to fit the model for X_train and y_train.

Then prediction is done for X_test using predict.

Theb the results of predicted value and y_test are compared.

Confusion matrix is plotted using the fuction

Accuracy, Precision and Recall of model is shown using the in-built function

Improving Logistic Model
The data is standadised to [0,1] range and again trained anf tested the model.

Second improvement is done by changing the weights and running the model with new weights obtained from the function

DecisionTree
Here first, the dataset is split into Test and Train data (80-20%)

Further, from the Trian and Test data, diagnosis column is seperated.

There will be 4 datatsets now. X_train, y_train, X_test,y_test. Dataset with X will have all features and dataset y will have diagnosis column only.

Decision Tree function is used to fit the model for X_train and y_train.

Then prediction is done for X_test using svc_predict.

Theb the results of predicted value and y_test are compared.

Confusion matrix is plotted using the fuction

Accuracy, Precision and Recall of model is shown using the in-built function

Improving Decision tree model
Hyper parameters are changed to get the optimised parameters by using the scikit learn function

The model is trained and tested with the new models

ROC curve of best 3 models
To compare the results of all 3 models, I created a loop to run the 3 best models. With this resukts, I plotted a single ROC-AUC curve

References
https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

https://journals.plos.org/plosone/article/figures?id=10.1371/journal.pone.0140362

https://towardsdatascience.com/real-world-implementation-of-logistic-regression-5136cefb8125

https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

https://towardsdatascience.com/logistic-regression-using-gradient-descent-optimizer-in-python-485148bd3ff2

https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3


```python

```
