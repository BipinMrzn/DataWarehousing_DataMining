import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Create the data points
data = np.array([[5,7], [8,4], [3,3], [4,4], [3,7], [6,7], [6,1], [5,5]])

# Define DBSCAN parameters
eps = 3.5
min_samples = 4

# Create DBSCAN object and fit the data
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data)

# Identify core points, border points, and outliers
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Visualize the clusters
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Outliers are plotted as black
        col = 'black'

    class_member_mask = (labels == k)

    # Plot core points as larger markers
    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgewidth=2, markersize=14)

    # Plot border points as smaller markers
    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgewidth=2, markersize=6)

plt.title('DBSCAN clustering results (eps=3.5, min_samples=4)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Print core points
print("Core points:")
for i in dbscan.core_sample_indices_:
    print(data[i])

# Print boundary points
print("\nBoundary points:")
for i in range(len(labels)):
    if labels[i] != -1 and i not in dbscan.core_sample_indices_:
        print(data[i])

# Print outliers
print("\nOutliers:")
for i in range(len(labels)):
    if labels[i] == -1:
        print(data[i])