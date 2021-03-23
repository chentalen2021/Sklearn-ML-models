from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd

x = [1,2,3,4,5,6,7]
y = [0,0,6,6,5,1,2]

df = pd.DataFrame({
    'A':x,
    'B':y
})

I = df.iloc[:,0]
O = df.iloc[:,1]

I_train,I_test,O_train,O_test = train_test_split(I,O,test_size=0.2,random_state=0)

RT = DecisionTreeRegressor(max_depth=7,min_samples_leaf=0.1,random_state=0)

RT.fit(I_train,O_train)
pred = RT.predict(I_test)

print(pred)