import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [2, 2, 8, 5, 7, 6]
y = [10, 5, 4, 8, 5, 4]

data = list(zip(x, y))
print(data)

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)


plt.show()