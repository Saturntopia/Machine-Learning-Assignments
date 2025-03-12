import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title(title)
    plt.show()

# DBSCAN excels
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

# DBSCAN struggles
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)
X_blobs = StandardScaler().fit_transform(X_blobs)

# Clustering algorithms
def apply_clustering(X, dataset_name):
    dbscan = DBSCAN(eps=0.3, min_samples=5).fit(X)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    hierarchical = AgglomerativeClustering(n_clusters=3).fit(X)
    
    plot_clusters(X, dbscan.labels_, f"DBSCAN on {dataset_name}")
    plot_clusters(X, kmeans.labels_, f"k-Means on {dataset_name}")
    plot_clusters(X, hierarchical.labels_, f"Hierarchical Clustering on {dataset_name}")

# Run clustering on datasets
apply_clustering(X_moons, "Moons Dataset")
apply_clustering(X_blobs, "Blobs Dataset")