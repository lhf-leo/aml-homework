import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


# science2k_doc_word = np.load("science2k-doc-word.npy")
# kmeans = KMeans(n_clusters = 3, init = "random").fit(science2k_doc_word)
# label = kmeans.labels_
# lookup = {}
with open("science2k-titles.txt", ) as f:
    for line in f.readlines():
        print(line)

# inputMatrix = [row for row in science2k_doc_word]
# for row in kmeans.cluster_centers_:
#     inputMatrix.append(row)
# res = euclidean_distances(inputMatrix)[-3:, :]
# for row in res:
#     list = []
#     for i, num in enumerate(row):
#         if num in sorted(row)[:13]: list.append(i)
#     print(list)
#
# dists = euclidean_distances(kmeans.cluster_centers_, science2k_doc_word)
# closest = [sorted(range(len(dist)), key=lambda i: dist[i])[:10] for dist in dists]
# print(closest)

# print(science2k_doc_word)

# for num_cluster in range(2, 25, 3):
#     print("----------------------------------")
#     print("number of cluster is", num_cluster)
#     for _ in range(10):
#         kmeans = KMeans(n_clusters = num_cluster, init = "random").fit(science2k_doc_word)
#         print(kmeans.inertia_)


# X = science2k_doc_word
# distortions = []
# K = range(1, 20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()