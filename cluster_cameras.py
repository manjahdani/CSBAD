import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# ---- CONFIG ----
CSV_FILE = "cross_performance.csv"
MIN_THRESHOLD = 0.4
MAX_THRESHOLD = 5
STEP = 0.05
MAX_CLUSTERS_DISPLAY = 9

# ---- LOAD DATA ----
df = pd.read_csv(CSV_FILE, index_col=0)
mat = df.values
cam_names = df.index.tolist()

# ---- MAKE DISSIMILARITY MATRIX ----
# Since your matrix is a similarity, convert to dissimilarity:
dissimilarity = 1 - mat
dissimilarity = (dissimilarity + dissimilarity.T) / 2  # enforce symmetry

# ---- CONDENSE FOR LINKAGE ----
condensed = squareform(dissimilarity, checks=False)

# ---- HIERARCHICAL CLUSTERING ----
Z = linkage(condensed, method='average')

def calculate_clusters(linkage_matrix, min_threshold, max_threshold, step):
    first_occurrences = {}
    seen_clusters = set()
    for threshold in np.arange(min_threshold, max_threshold, step):
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        num_clusters = len(np.unique(clusters))
        if num_clusters not in seen_clusters:
            seen_clusters.add(num_clusters)
            first_occurrences[num_clusters] = {
                'threshold': threshold,
                'clusters': []
            }
            for cluster_num in range(1, num_clusters + 1):
                indices = np.where(clusters == cluster_num)[0]
                first_occurrences[num_clusters]['clusters'].append(indices)
    return first_occurrences

def print_clusters(clusters_dict, cam_names, max_clusters=9):
    sorted_keys = sorted(clusters_dict.keys())
    for num_clusters in sorted_keys[:max_clusters]:
        v = clusters_dict[num_clusters]
        print(f"{num_clusters} clusters at threshold {v['threshold']:.2f}:")
        for i, idxs in enumerate(v['clusters']):
            names = [cam_names[idx] for idx in idxs]
            print(f"  Cluster {i+1}: {names}")

if __name__ == "__main__":
    clusters = calculate_clusters(Z, MIN_THRESHOLD, MAX_THRESHOLD, STEP)
    print_clusters(clusters, cam_names, MAX_CLUSTERS_DISPLAY)
