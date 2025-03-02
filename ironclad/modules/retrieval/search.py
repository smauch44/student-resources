import numpy as np

class FaissSearch:
    def __init__(self, faiss_index, metric='euclidean'):
        """
        Initialize the search class with a FaissIndex instance and distance metric.
        
        :param faiss_index: A FaissIndex instance.
        :param metric: The distance metric ('euclidean', 'dot_product', 'cosine', 'minkowski').
        :param p: The parameter for Minkowski distance (p=2 for Euclidean, p=1 for Manhattan).
        """
        self.index = faiss_index.index
        self.metric = metric
        
        self.faiss_index = faiss_index

    def search(self, query_vector, k=5 ,p=3):
        """
        Perform a nearest neighbor search and retrieve the associated metadata.
        
        :param query_vector: The vector to query (numpy array).
        :param k: Number of nearest neighbors to return.
        :param p: Optional Minkowski distance parameter.
        :return: Distances, indices, and metadata of the nearest neighbors.
        """
        if self.metric == 'euclidean':
            # Default FAISS search (Euclidean)
            distances, indices = self.index.search(query_vector, k)

        elif self.metric == 'cosine':
            pass
        elif self.metric == 'dot_product':
            pass
        elif self.metric == 'minkowski':
            pass


if __name__ == "__main__":
    import numpy as np

    from index.bruteforce import FaissBruteForce


    # Create some random vectors (10k vectors of dimension 128)
    vectors = np.random.random((10000, 256)).astype('float32')
    metadata = [f"Vector_{i}" for i in range(10000)]
    query_vector = np.random.random((1, 256)).astype('float32')
    # Construct the Brute Force Index
    faiss_index_bf = FaissBruteForce(dim=256)

    print("\nExample 1: BruteForce Search with `euclidean` measure")
    faiss_index_bf.add_embeddings(vectors, metadata=metadata)
    search_euclidean = FaissSearch(faiss_index_bf, metric='euclidean')
    distances, indices, metadata = search_euclidean.search(query_vector, k=5)
    for i in range(5):
        print(f"Nearest Neighbor {i+1}: Index {indices[0][i]}, Distance {distances[0][i]}, Metadata: {metadata[i]}")

    print("\nExample 2: BruteForce Search with `cosine` measure")
    search_cosine = FaissSearch(faiss_index_bf, metric='cosine')
    distances, indices, metadata = search_cosine.search(query_vector, k=5)
    for i in range(5):
        print(f"Nearest Neighbor {i+1}: Index {indices[0][i]}, Distance {distances[0][i]}, Metadata: {metadata[i]}")