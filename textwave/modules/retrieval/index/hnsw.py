import faiss
import numpy as np
import pickle


class FaissHNSW:
    """
    A FAISS HNSW index for storing embeddings and their associated metadata using
    the Hierarchical Navigable Small World (HNSW) algorithm.

    This class leverages FAISS's HNSW index for similarity search in large datasets.
    HNSW indexes provide fast approximate nearest neighbor search without the need for a
    separate training step. The index directly adds embeddings, partitions them into a graph-based
    structure, and uses this structure to perform rapid similarity queries.

    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list storing metadata associated with each embedding.
        index (faiss.IndexHNSWFlat): The underlying FAISS HNSW index object.
    """

    def __init__(self, dim, M=32, efConstruction=200):
        """
        Initializes the FaissHNSW instance.

        This method creates a FAISS HNSW index for flat (L2) search. It accepts keyword arguments
        to configure parameters of the HNSW algorithm, including:
          - M (int): The number of neighbors in the HNSW graph. A higher value leads to a more connected graph
                     and can improve search quality at the expense of increased memory usage. Default is 32.
          - efConstruction (int): The size of the dynamic candidate list during the index construction phase.
                                  Larger values improve index quality but increase construction time. Default is 40.

        After initialization, the index is ready to accept embeddings without a training step.

        Parameters:
            dim (int): The dimensionality of the embeddings.
            **kwargs: Optional keyword arguments to configure the HNSW index. Recognized keys include 'M' and 'efConstruction'.
        """
        self.dim = dim
        self.metadata = []
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = efConstruction

    def add_embeddings(self, embeddings, metadata):
        """
        Adds new embeddings and their associated metadata to the HNSW index.

        This method processes each new embedding by converting it into a NumPy array and verifying
        its dimensionality. If an embedding's dimension does not match the expected dimension, a
        ValueError is raised. The corresponding metadata for each embedding is stored in an internal
        list. The embeddings are reshaped to a two-dimensional array before being added to the index.

        Parameters:
            new_embeddings (list or np.ndarray): A list or array of new embeddings to be added.
                Each embedding should be an array-like object with a length equal to `dim`.
            new_metadata (list): A list of metadata items corresponding to each embedding.

        Raises:
            ValueError: If the number of embeddings does not match the number of metadata entries.
            ValueError: If any individual embedding does not match the specified dimensionality.
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        # Validate dimensions
        if len(embeddings.shape) != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings should have shape (N, {self.dim}), but got {embeddings.shape}.")

        if len(embeddings) != len(metadata):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match metadata ({len(metadata)}).")

        # Add embeddings and metadata
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def get_metadata(self, idx):
        """
        Retrieves the metadata associated with a specific embedding based on its index.

        The given index corresponds to the position in the metadata list where the metadata was stored
        when the embedding was added. This method validates the index and raises an IndexError if it is out of bounds.

        Parameters:
            idx (int): The integer index of the embedding for which metadata is being requested.

        Returns:
            The metadata object associated with the specified embedding.

        Raises:
            IndexError: If the provided index is negative or exceeds the number of stored embeddings.
        """
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError("Index out of bounds.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Serializes and saves the FaissHNSW instance to a file using pickle.

        This method saves the entire state of the instance, including the FAISS index, metadata,
        and configuration parameters (dim, M, and efConstruction). The saved file can later be loaded
        to restore the index for future use without needing to reinitialize or rebuild.

        Parameters:
            filepath (str): The file path where the serialized instance will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a serialized FaissHNSW instance from a file.

        This class method reads the pickle file at the given file path and returns a FaissHNSW
        instance with its state restored. This allows users to resume operations on a previously
        saved index.

        Parameters:
            filepath (str): The file path from which the instance will be loaded.

        Returns:
            An instance of FaissHNSW with the state restored from the specified file.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


# Example usage of FaissHNSW class.
if __name__ == "__main__":

    # Initialize a FaissHNSW index with embedding dimension 4 using custom HNSW parameters.
    index = FaissHNSW(dim=4, M=16, efConstruction=50)

    # Create some dummy embeddings and corresponding metadata.
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    identity_metadata = [
        "Alice",
        "Bob",
        "Charlie"
    ]

    # Add the embeddings and metadata to the index.
    index.add_embeddings(embeddings, identity_metadata)

    # Let's search the index with a query vector.
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    k = 2  # number of nearest neighbors to retrieve

    # The FAISS search directly returns distances and indices.
    distances, indices = index.index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]

    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_hnsw_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissHNSW.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
