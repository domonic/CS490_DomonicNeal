# CS490_DomonicNeal


In this lesson we learned about clustering and its importance as it pertains to clustering data based on information contained 
with the data itself. We were able to use and implement the KMeans algorithm to cluster a given data set and then perform 
dimensionality reduction on the same data before performing the KMeans algorithm again which ultimately improved our 
Silhouette Score which measures how closely related data points are to one another within their own cluster oppose to other clusters. 


Task 1:
Apply K means clustering in this data set provided below:
Remove any null values by the mean.
Use the elbow method to find a good number of clusters with the KMeans algorithm


Task 2:
Calculate the silhouette score for the above clustering


Task 3:
Apply PCA on the same dataset


**BONUS:
Apply kmeans algorithm on the PCA result and report your observation if the score improved or not?

After observing the two scores performing PCA and then using the KMeans Algorithm on the data again the score did imporve and
was closer to 0 because of the scaling done with PCA oppose to simply performing KMeans Algorithm on the data which resulted in a lower score than the PCA.
