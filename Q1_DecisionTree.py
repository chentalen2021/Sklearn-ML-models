import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix,roc_curve,auc,mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier

#Read the data file and transform into dataframe
df_Forest = pd.read_excel(r'/Users/talen/Downloads/Forest.xlsx')

print(df_Forest.columns)
#Check the dataframe to see whether it has missing values and the data type of each column
#print(df_Forest.isnull().sum())
#print(df_Forest.dtypes)

#Split the dataframe into predictors and target columns
predictors = df_Forest.iloc[:,1:]
target = df_Forest.iloc[:,0]

#Split the predictor and target sets into training and testing sets
predi_train, predi_test, target_train, target_test = train_test_split(predictors, target, test_size=0.3,random_state=100)

#Enlabel the non-numeric column
class_le = LabelEncoder()
target_train = class_le.fit_transform(target_train)
target_test = class_le.transform(target_test)

#Standardise the predictors for Decision Tree classifier
scaler1 = StandardScaler()
predi_train_tree = scaler1.fit_transform(predi_train)
predi_test_tree = scaler1.transform(predi_test)
    #Standardise the whole predictors for cross-validation
predictors_tree = scaler1.fit_transform(predictors)

#Normalise the predictors for MultiLayer Perceptron
scaler2 = MinMaxScaler()
predi_train_MLP = scaler2.fit_transform(predi_train)
predi_test_MLP = scaler2.transform(predi_test)
    #Normalise the whole predictors for cross-validation
predictors_MLP = scaler2.fit_transform(predictors)

#Q1 Sub-question 1
#Build Decision Tree classifier and make predictions
    #Determine the best max_depth parameter for Decision Tree by mean-square-error
accuracies=[]
depths=[]
samples_split=[]
samples_leaf =[]

for x in range(1,11):
    for y in range(10,50):
        for z in range(5,50):
            clf_tree_try = DecisionTreeClassifier(criterion='entropy',max_depth=x,min_samples_split=y,min_samples_leaf=z,random_state=100)
            clf_tree_try.fit(predi_train_tree,target_train)
            predictions_tree = clf_tree_try.predict(predi_test_tree)
            accuracies.append(cross_val_score(clf_tree_try,predictors_tree,target,cv=3).mean())
            depths.append(x)
            samples_split.append(y)
            samples_leaf.append(z)

df_param_accu = pd.DataFrame({
    'Depth' : depths,
    'Min_samples_split' : samples_split,
    'Min_samples_leaf' : samples_leaf,
    'Accurarcy' : accuracies
})

df_param_accu.to_csv('df_param_accu.csv')


'''
    #Based on the graph in the above section, the best max_depth of the Decision Tree classifier is 4, now build the tree
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=100)
clf_tree.fit(predi_train_tree, target_train)
predictions_tree = clf_tree.predict(predi_test_tree)

#Build Multilayer Perceptron and make predictions
clf_MLP = MLPClassifier(random_state=100)
clf_MLP.fit(predi_train_MLP,target_train)
predictions_MLP = clf_MLP.predict(predi_test_MLP)

#Plot Confusion Matrices for Decision Tree and MLP classifiers and calculate their accuracies
    #For Decision Tree
plot_conmatrix_tree = plot_confusion_matrix(clf_tree,predi_test_tree,target_test)
plot_conmatrix_tree.ax_.set_title('Confusion Matrix for Decision Tree')

print('The accuracy of Decision Tree classifier is: ',accuracy_score(target_test,predictions_tree))
print("The average accuracy score of Decision Tree under cross validation is: ",cross_val_score(clf_tree,predictors_tree,target,cv=3).mean())

    #For MLP
plot_conmatrix_MLP = plot_confusion_matrix(clf_MLP,predi_test_MLP,target_test)
plot_conmatrix_MLP.ax_.set_title('Confusion Matrix for Multilayer Perceptron')

print('The accuracy of Multilayer Perceptron classifier is: ',accuracy_score(target_test,predictions_MLP))
print("The average accuracy score of MLP under cross validation is: ",cross_val_score(clf_MLP,predictors_MLP,target,cv=3).mean())

plt.show()

#Q1 Sub-question 2
    #Generate probability table for both classifiers
prob_tree = clf_tree.predict_proba(predi_test_tree)
prob_MLP = clf_MLP.predict_proba(predi_test_MLP)

    #Probability table for the first sample
probability_table_sample1 = pd.DataFrame([[prob_tree[0][0],prob_MLP[0][0]],
                                 [prob_tree[0][1],prob_MLP[0][1]],
                                 [prob_tree[0][2],prob_MLP[0][2]],
                                 [prob_tree[0][3],prob_MLP[0][3]]],
                                 columns=['Decision Tree','MLP'], index=['Class_d','Class_h','Class_o','Class_s'])
print(probability_table_sample1)

#Q1 Sub-question 4
    #Determine the class of the i-th sample

def clf_DT_MLP_Aggregate(i):
    #create a DataFrame containing all the samples and their probabilities in each class with regard to each classifier
    samples = []
    class_b_proba = []
    class_h_proba = []
    class_o_proba = []
    class_s_proba = []
    for a in range(len(target_test)):
        samples.append(a)
        samples.append(a)
        class_b_proba.append(prob_tree[a][0])
        class_b_proba.append(prob_MLP[a][0])
        class_h_proba.append(prob_tree[a][1])
        class_h_proba.append(prob_MLP[a][1])
        class_o_proba.append(prob_tree[a][2])
        class_o_proba.append(prob_MLP[a][2])
        class_s_proba.append(prob_tree[a][3])
        class_s_proba.append(prob_MLP[a][3])

    proba_table_DT_MLP = pd.DataFrame({
        'Samples': samples,
        'Class_b_proba' : class_b_proba,
        'Class_h_proba' : class_h_proba,
        'Class_o_proba' : class_o_proba,
        'Class_s_proba' : class_s_proba
    })

    #Use Aggregate and Average functions to generate a new table with the average probability of the two classifiers in each class grouped by 'Samples'
    proba_table_average = proba_table_DT_MLP.groupby('Samples').aggregate([np.average])
    #Reversely index the column number by the values and this column number is the class ID
    for y in range(4):
        if proba_table_average.iloc[i,y] == proba_table_average.max(axis=1)[i]:
            final_class = y
            #print('The No.{0} sample belongs to class {1}'.format(i,final_class))
    return final_class

predictions_DT_MLP = []
for i in range(len(target_test)):
    predictions_DT_MLP.append(clf_DT_MLP_Aggregate(i))

accuracy_classifier_DT_MLP = accuracy_score(target_test,predictions_DT_MLP)
print('The accuracy of the Tree-MLP-combined classifier is: ',accuracy_classifier_DT_MLP)


#Q1 Sub-question 5
# Pr(class=’s’|DT=’s’) is actually the precision of predicting class 's' by Decision Tree; Get all the precision scores of all the class-predictions
from sklearn.metrics import classification_report
report = classification_report(target_test,predictions_tree,output_dict=True)

precisions = {
    0:report['0']['precision'],
    1:report['1']['precision'],
    2:report['2']['precision'],
    3:report['3']['precision']
}
print(precisions)

def classifier_DT_MLP_conditional(i):
    samples = []
    classes =[]
    Proba_DT = []
    Proba_MLP = []

    samples.append(i)
    samples.append(i)
    samples.append(i)
    samples.append(i)
    classes.append(0)
    classes.append(1)
    classes.append(2)
    classes.append(3)
    Proba_DT.append(prob_tree[i][0])
    Proba_DT.append(prob_tree[i][1])
    Proba_DT.append(prob_tree[i][2])
    Proba_DT.append(prob_tree[i][3])
    Proba_MLP.append(prob_MLP[i][0])
    Proba_MLP.append(prob_MLP[i][1])
    Proba_MLP.append(prob_MLP[i][2])
    Proba_MLP.append(prob_MLP[i][3])

    proba_table_condition = pd.DataFrame({
        'Samples' : samples,
        'Classes' : classes,
        'Proba_DT' : Proba_DT,
        'Proba_MLP' : Proba_MLP
    })

    for a in range(4):
        if proba_table_condition.iloc[a,2]==proba_table_condition.max()[2]:
            max_class_DT = a

    for b in range(4):
        if proba_table_condition.iloc[b,3]==proba_table_condition.max()[3]:
            max_class_MLP = b
    
    P1 = proba_table_condition.max()[2] * precisions[max_class_DT]
    P2 = proba_table_condition.max()[3] * precisions[max_class_MLP]

    if P1 > P2: return max_class_DT
    else: return max_class_MLP

id = int(input('Input the sample_ID you want to predict its class: '))
print('The No.{0} sample belongs to class {1}'.format(id,classifier_DT_MLP_conditional(id)))

'''