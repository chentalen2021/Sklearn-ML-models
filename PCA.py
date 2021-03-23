import pandas as pd
import numpy as np
import random as rd

a = ['n' + str(i) for i in range(1,6)]
b = ['m' + str(t) for t in range(1,6)]

genes = ['gene' + str(i) for i in range(1,101)]

df = pd.DataFrame(columns=[*a,*b],index=genes)

for gene in df.index:
    df.loc[gene,'a1':'a5'] = np.random.poisson(lam=rd.randrange(10,1000),size=5)
    df.loc[gene,'b1':'b5'] = np.random.poisson(lam=rd.randrange(10,1000),size=5)


print(df.head())
print(df.shape)


from sklearn.preprocessing import scale
#The dataFrame should be scaled before applying PCA
#The scale will make the values in the dataFrame have a mean value of 0, and standard deviation of 1
scaled_data = scale(df.T)   #Transpose the data to make the rows decode the genes