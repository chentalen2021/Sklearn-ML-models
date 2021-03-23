import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_forest = pd.read_excel(r'/Users/talen/Downloads/Other Datasets/Forest.xlsx')

#Checking whether there is missing data in the dataframe
print(df_forest.isnull().sum().sum())


#Use the first column — class as the result, whereas use the rest columns as the properties
X = df_forest.iloc[:,1:]
Y = df_forest.iloc[:,0]

#Split the dataset into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state=0)

#Using Decision Tree Classifier to train the dataset and make prediction
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

tree_clf = DecisionTreeClassifier(criterion='entropy', random_state=70, max_depth=4,min_samples_leaf=5)
tree_clf.fit(x_train,y_train)

tree_pred = tree_clf.predict(x_test)

#Plot the tree
tree.plot_tree(tree_clf, filled=True, class_names=df_forest.columns)

#Using Confusion Matrix to evaluate the accuracy of the prediction made by Decision Tree Method
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test,tree_pred)

tree_plot = plot_confusion_matrix(tree_clf,x_test,y_test)
tree_plot.ax_.set_title('Confusion Matrix for Decision Tree')

plt.show()

accuracy_tree = accuracy_score(y_test,tree_pred)
print(accuracy_tree)

