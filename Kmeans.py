import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
print(sys.executable)


data = np.random.rand(1000, 2) * 100

km = KMeans(n_clusters=3, init="random")

t0 = time.process_time()
km.fit(data)
t1 = time.process_time()

tt = t1 - t0
print("Total Time:", tt)

centers = km.cluster_centers_
labels = km.labels_

print("Cluster Centers:", centers)
# print("Cluster Labels:", *labels)

colors = [ "b", "y", "m"]
markers = ["+", "*",  "d"]

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], color=colors[labels[i]], marker=markers[labels[i]])
plt.scatter(centers[:, 0], centers[:, 1], marker="o", s=50, linewidths=5)
plt.show()