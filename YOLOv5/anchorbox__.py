import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the train.json file
with open('/data/ephemeral/home/jiwan/dataset/train.json', 'r') as f:
    data = json.load(f)

# Extract bounding box dimensions (width and height)
bbox_dims = []
for annotation in data['annotations']:
    x, y, width, height = annotation['bbox']  # x, y, width, height
    bbox_dims.append([width, height])

# Convert list to numpy array
bbox_dims = np.array(bbox_dims)

# Define the number of clusters (the number of anchor boxes you want)
num_clusters = 9  # For example, you can change this to the number of anchors you prefer

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(bbox_dims)

# Get the anchor boxes (cluster centers)
anchors = kmeans.cluster_centers_

# Sort anchors by area (optional, for better readability)
anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

print("Optimal anchor boxes (width, height):")
print(anchors)

# Optionally, plot the clusters and anchor boxes
plt.scatter(bbox_dims[:, 0], bbox_dims[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(anchors[:, 0], anchors[:, 1], color='black', marker='x')
plt.title('K-means Clustering of Bounding Box Dimensions')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()
