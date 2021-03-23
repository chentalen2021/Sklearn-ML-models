import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings(action='ignore')

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
rawdata = pd.read_csv(url, names=names)
array = rawdata.values
nrow, ncol = rawdata.shape
X = array[:, 0:8]
Y = array[:, 8]

def get_accuracy(target_tain, target_test,predicted_test,predicted_train):
    clf = MLPClassifier(activation='logistic',solver='sgd',learning_rate_init=0.1,alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
    clf.fit(predicted_train,np.ravel(target_tain,order='C'))
    predictions = clf.predict(predicted_test)
    return accuracy_score(target_test,predictions)

pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=.3, random_state=4)
print("Accuracy score of our model without feature selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test, pred_train))

'''
#Chi Square 
test = SelectKBest(score_func=chi2,k=3)
fit = test.fit(X,Y)
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

print(features[0:3, :])
print()

features = pd.DataFrame(features)
print(features)

pred_features = features

pred_train,pred_test,tar_train,tar_test = train_test_split(pred_features,Y,test_size=0.3,random_state=1)
print("Accuracy score of our model with chi square feature selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
'''
'''
#Recursive Feature Extraction (RFE)
model = LogisticRegression()
rfe = RFE(model,n_features_to_select=3)
fit = rfe.fit(X,Y)
print("Num Features: %d" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

features = fit.transform(X)
pred_features = features
print(features)
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with RFE selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()
print(rawdata)
'''

#Principal Components Analysis (PCA)
pca = PCA(n_components=5)
fit = pca.fit(X)

#print("Explained Variance: %s" % (fit.explained_variance_ratio_))
#print(fit.components_)

features = fit.transform(X)
print(features)
'''
pred_features = features
pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
print("Accuracy score of our model with PCA selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
print()
'''
