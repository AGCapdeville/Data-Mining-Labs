'''
Lab 9
'''
print ("Lab 9")
######### IMPORTS #########
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from sklearn.mixture import GaussianMixture
from scipy.stats import linregress
import math

##########Part 0 ###########
'''
    1)  load iris-data.csv (the original dataset) using Pandas and save it as a dataframe (data)
        drop "species" column form your dataframe and store it in a new dataframe (Y)
        Hint: pandas.DataFrame.drop() to drop unwanted features
'''
# YOUR CODE GOES HERE
print("\n======================================\n\t\tPART 0\n======================================")
with open('./data/iris-data.csv') as csvfile:
    iris_data = pd.read_csv(csvfile, delimiter=',')

species = iris_data['species']
iris_data.drop(columns="species", inplace=True)
# print(data)

##########Part 1 ###########
'''
    1)  Now the data can be used for clustering task because you have removed the labels. This dataset has 4 features: 'sepal_length',  'sepal_width',  'petal_length',  'petal_width'.
        try to cluster your data using Kmeans
        set: random_state=987  and n_clusters=3 (here we can use our prior information about the data ( the original dataset has 3 classes))
'''
# YOUR CODE GOES HERE

print("\n======================================\n\t\tPART 1\n======================================")
k_means = KMeans(random_state=987, n_clusters=3).fit(iris_data)

'''
    2)  plot the three clusters in 2D graphs (Cluster1: green , cluster2: red, cluster3:blue)
'''
# YOUR CODE GOES HERE
def show_data(feature_col, feature_row, dataframe, k_means):
    centroids = k_means.cluster_centers_
    plt.scatter(feature_col, feature_row, c= k_means.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.show()

# show_data(iris_data['sepal_length'], iris_data['sepal_width'], iris_data, k_means)

'''
    3)  Use the actual lables to calculate the accuracy of the resulted lables from your K-Means model.
'''
# YOUR CODE GOES HERE
def get_labels_as_num_list(labels, target_names):
    target = []
    for l in labels:
        for index in range(len(target_names)):
            if target_names[index] == l:
                target.append(index)
    return target

target_names = ['virginica', 'setosa', 'versicolor']
targets = get_labels_as_num_list(species, target_names)
# print(classification_report(targets, k_means.labels_, target_names=target_names))

'''
    4)  In general there is no global theoretical method to find the optimal value of K in K-means, but one common technique to determine optimal K is "elbow" method:
        Do a quick reasech about this method, and briefly explain it here.
        write a python code to find the optimal k using "elbow" method
'''
# YOUR CODE GOES HERE
# Elbow method: Explanation...
# The Elbow methods main idea is to run k-means clustering for a range of clusters k (let’s say from 1 to 10) and for each value, we are calculating the sum of squared distances 
# from each point to its assigned center(distortions). When the distortions are plotted and the plot looks like an arm then the “elbow”(the point of inflection on the curve) 
# is the best value of k.

# We found the best K based off of distance from an imaginary line made by the first and last points. Instead of observing the "Elbow" we found the point in the "Elbow" through 
# finding the furthest point. 

k_times = 10
distortions = []
k_range = range(1, k_times + 1)
for k in k_range:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(iris_data)
    distortions.append(kmeanModel.inertia_)

plt.plot(k_range, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

rise = distortions[0]
run = k_times
slope = -(rise / float(run))

dist_list = []
for index in range(0, len(distortions)):
    imaginaryLineY = slope * (index) + rise
    distance = abs(distortions[(index)] - imaginaryLineY)
    dist_list.append(distance)

plt.plot(dist_list)
plt.show()
print("These are the distances: ", dist_list)
print("Therefore the optimal K:", dist_list.index(max(dist_list))+1, '\n')


##########Part 2 ###########
'''
    1)  try to cluster your data using Agglomerative Clustering
        set: n_clusters=3 
        try different linkage method and pick the most accurate one (Hint: calculate accuracy)
'''
# YOUR CODE GOES HERE
print("\n======================================\n\t\tPART 2\n======================================")
linkage_list = ['single', 'ward', 'average', 'complete']
accuracy_list = []
for link in linkage_list:
    clustering = AgglomerativeClustering(n_clusters= 3, linkage= link).fit(iris_data)
    report = classification_report(targets, clustering.labels_, target_names=target_names, output_dict=True)
    accuracy_list.append(report['accuracy'])

linkage_method = linkage_list[accuracy_list.index(max(accuracy_list))]
print("Most Accurate method is \"",linkage_method,"\" of Accuracy:", max(accuracy_list))

'''
    2)  plot the dendrogram for the selected clustering above
'''
# YOUR CODE GOES HERE
# hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
# linkage = hierarchy.linkage(iris_data, linkage_method)
# plt.figure()
# dn = hierarchy.dendrogram(linkage)
# plt.show()

##########Part 3 ###########
'''
    1)  try to cluster your data using GMM and calculate the accuracy
        set: n_components=3
'''
# YOUR CODE GOES HERE
# gmm = GaussianMixture(n_components=3).fit(iris_data)
# labels = gmm.predict(iris_data)
# plt.scatter(iris_data['sepal_length'], iris_data["sepal_width"],c=labels, cmap='viridis')
# plt.show()