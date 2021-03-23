from IPython.display import Image
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

#Create a sample data set using make_blobs. This particular dataset has 2 features and 3 clusters.
X,Y,Center = make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0, return_centers=True)

plt.figure(1)
plt.scatter(X[:,0],X[:,1], c='white', marker='o', edgecolors='black',s=50)

#Apply K-means clustering with 3 centroids
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    #We set n_init=10 to run the k-means clustering algorithms 10 times independently
    #with different random centroids to choose the final model as the one with the lowest SSE

#Predicting cluster labels
predictions = km.fit_predict(X)

#Visualize the clusters identified(using y_km)together with cluster labels.
plt.figure(2)
plt.scatter(X[predictions==0,0],X[predictions==0,1],s=50, c='lightgreen', marker='s', edgecolors='black',label='cluster1')
plt.scatter(X[predictions==1,0],X[predictions==1,1],s=50, c='orange', marker='o', edgecolors='black', label='cluster2')
plt.scatter(X[predictions==2,0],X[predictions==2,1],s=50, c='lightblue', marker='v', edgecolors='black', label='cluster3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],s=250,marker='*',c='red', edgecolors='black', label='centroids')

plt.legend(scatterpoints = 1)

print(km.transform(X))

plt.show()

#Calculating Distortion
print('Distortion: %.2f' %km.inertia_)
'''
#Observing the behaviour of the distortion with the number of clusters.
#Using elbow method to find optimum number of clusters
distortions = []

for i in range(1,11):
    km1 = KMeans(n_clusters=i,init='k-means++', n_init=10, max_iter=300, random_state=0)
    km1.fit(X)
    distortions.append(km1.inertia_)

plt.figure(3)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')

#Quantifying the quality of clustering via silhouette plots.
#k-means++ gives better clustering/performance than classic approach(init=’random’).
cluster_labels = np.unique(predictions)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, predictions, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[predictions == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.figure(4)
plt.axvline(silhouette_avg, color = 'red', linestyle = '--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()

'''















