from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

df = pd.read_excel(r'/Users/talen/Downloads/Other datasets/Iris.xlsx')
print(df.shape)

'''
#Select one value for experiment
original_value = float(df.loc[20,['px_height']])
original_class = df.iloc[20,-1]
print('The original value of the missing data is: ', original_value)

df.loc[20,['px_height']] = np.nan

#Use the missing data's class to select the sub-DF
print('The class of the missing value is: ',original_class)


df_original_class = df[df.iloc[:,-1]==original_class]
#print('The shape of the dataframe with target class is: ',df_original_class.shape)


#Get the initial guess of the missing data based on the mean value of that column
guess_value = np.sum(df['px_height'])/(df_original_class.shape[0]-1)
print('The initial guess of the missing data is: ', guess_value)

#Assign the missing data to the original DF, and split the DF
df.loc[20,['px_height']] = guess_value
X= df.iloc[:,:-1]
Y= df.iloc[:,-1]

#Use the DF to build up Random Forest
clf = RandomForestClassifier(n_estimators=100,criterion='entropy',bootstrap=True,random_state=0)
clf.fit(X,Y)

#Apply the RF to the original dataset to get the leaf path of each sample
Decision_path_df = pd.DataFrame(clf.apply(X), index=('sample' + str(i) for i in range(df.shape[0])),
                                columns=('tree path ' + str(t) for t in range(100)))
print(Decision_path_df)

print('---------------------------------------------------------------')

from ProximityMatrix import ProximityMatrix
print('The proximity matrix is:\n',ProximityMatrix(Decision_path_df))

'''