from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error

X,Y = make_blobs(n_features=2, n_samples=100,random_state=1,shuffle=False)

km = KMeans(n_clusters=2)
labels = km.fit_predict(X)

print('The inertia is: ', km.inertia_)

#For SSE
    #For cluster 1
SSE1=0
for i in range(len(X[labels==0])):
    SSE1 += (km.cluster_centers_[0][0]-X[labels==0,0][i])**2 + (km.cluster_centers_[0][1]-X[labels==0,1][i])**2

SSE2=0
for t in range(len(X[labels==1])):
    SSE2 += (km.cluster_centers_[1][0]-X[labels==1,0][t])**2 + (km.cluster_centers_[1][1]-X[labels==1,1][t])**2

SSE = SSE1 + SSE2

print('The SSE is: ',SSE)