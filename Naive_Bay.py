from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df = pd.read_excel(r'/Users/talen/Downloads/Iris.xlsx')
predictor = df.iloc[:,:-1]
target = df.iloc[:,-1]

predictor_train,predictor_test,target_train,target_test = train_test_split(predictor,target,test_size=0.2,random_state=100)

gnb = GaussianNB()
gnb.fit(predictor_train,np.ravel(target_train,order='C'))
predictions = gnb.predict(predictor_test)
print("Accuracy score of our model with Gaussian Naive Bayes:", accuracy_score(target_test, predictions))

mnb = MultinomialNB()
mnb.fit(predictor_train,np.ravel(target_train,order='C'))
predictions = mnb.predict(predictor_test)
print("Accuracy score of our model with Multinomial Naive Bayes:", accuracy_score(target_test, predictions))