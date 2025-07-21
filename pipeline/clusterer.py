import numpy as np
import hdbscan

def cluster_embeddings(
    processed_data: list[dict],
    min_cluster_size: int = 5,
    min_samples: int | None = 1,
    metric: str = 'cosine',
    cluster_selection_epsilon: float = 0.2
) -> list[dict]:
    """
    Performs HDBSCAN clustering on a list of embeddings.
    """
    embeddings = [item['embedding'] for item in processed_data]
    if not embeddings:
        print("No embeddings found to cluster.")
        return []
    
    embedding_matrix = np.array(embeddings)

    print(f"Clustering {len(embedding_matrix)} embeddings with HDBSCAN...")

    # 如果块的数量太少，无法进行有意义的聚类，则将所有块分配给一个簇
    if len(embedding_matrix) < min_cluster_size:
        print(f"Number of embeddings ({len(embedding_matrix)}) is less than min_cluster_size ({min_cluster_size}). Assigning all to cluster 0.")
        cluster_labels = np.zeros(len(embedding_matrix), dtype=int)
    else:
        print(f"Params: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='{metric}', method='leaf', epsilon={cluster_selection_epsilon}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            algorithm='generic',
            cluster_selection_method='leaf',
            cluster_selection_epsilon=cluster_selection_epsilon,
            gen_min_span_tree=True
        )
        
        cluster_labels = clusterer.fit_predict(embedding_matrix)

    for i, item in enumerate(processed_data):
        item['cluster_id'] = int(cluster_labels[i])
        # Remove embedding to save memory as it's not needed for the next step
        del item['embedding']

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)
    print(f"Clustering complete. Found {num_clusters} clusters and {num_noise} noise points.")

    return processed_data