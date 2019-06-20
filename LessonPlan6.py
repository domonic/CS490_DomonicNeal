import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings

warnings.filterwarnings("ignore")


'''Read the data from the csv file '''
data_set = pd.read_csv('./CC.csv')

'''Replace Null Values'''
mean_value = data_set.mean()
data_set = data_set.replace(0, np.nan)
data_set = data_set.fillna(mean_value)

data_set = data_set.apply(LabelEncoder().fit_transform)

'''Independent variables (features)'''
data_set_x = data_set.iloc[:, 0: 18]

scaler = preprocessing.StandardScaler()
scaler.fit(data_set_x)
X_scaled_array = scaler.transform(data_set_x)
X_scaled = pd.DataFrame(X_scaled_array, columns=data_set_x.columns)

'''Use the elbow method to find a good number of clusters with the KMeans algorithm'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_set_x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


'''Based on Elbow Method it was determined number of clusters should be 3'''
num_clusters = 4

km = KMeans(n_clusters=num_clusters)
km.fit(data_set_x)

y_cluster_kmeans = km.predict(data_set_x)
score = metrics.silhouette_score(data_set_x, y_cluster_kmeans)

print("Silhouette Score:", score)

'''Apply PCA on the dataset'''

pca_scaler = StandardScaler()
'''Fit on training set only.'''
pca_scaler.fit(data_set_x)

'''Application of transform the training and test set.'''
x_scaler = pca_scaler.transform(data_set_x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, data_set[['TENURE']]], axis=1)
print(finaldf)

'''Use the elbow method to find a good number of clusters with the KMeans algorithm'''
pca_wcss = []
for i in range(1, 11):
    pca_kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pca_kmeans.fit(data_set_x)
    pca_wcss.append(pca_kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

'''Application of kmeans algorithm on PCA result'''
num_clusters_pca = 4

km_pca = KMeans(n_clusters=num_clusters_pca)
km_pca.fit(x_scaler)

y_cluster_kmeans_pca = km_pca.predict(x_scaler)
score_pca = metrics.silhouette_score(x_scaler, y_cluster_kmeans_pca)
print("Silhouette Score PCA:", score_pca)
