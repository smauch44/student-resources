import numpy as np
import pickle
import faiss


class FaissBruteForce:
    """
    A brute-force FAISS index for storing embeddings and their associated metadata.
    
    Attributes:
        dim (int): The dimensionality of the embeddings.
        embeddings (list): A list to store numpy array representations of embeddings.
        metadata (list): A list to store metadata corresponding to each embedding.
        index (faiss.IndexFlat): A FAISS flat index initialized with the specified embedding dimension.
    """

    def __init__(self, dim):
        """
        Initializes the FaissBruteForce index.
        
        This method sets up internal storage for embeddings and metadata and creates a FAISS flat index.
        
        Parameters:
            dim (int): The dimensionality of the embeddings.
        """
        self.dim = dim
        self.metadata = []    # Will store associated metadata.
        self.index = faiss.IndexFlat(dim)

    def add_embeddings(self, embeddings, metadata):
        """
        Adds new embeddings and their associated metadata to the index.
        
        Parameters:
            new_embeddings (list or np.ndarray): A list of embeddings, where each embedding is an array-like
                of length `dim`.
            new_metadata (list): A list of metadata corresponding to each embedding.
        
        Raises:
            ValueError: If an embedding does not match the specified dimensionality.
            ValueError: If the lengths of new_embeddings and new_metadata do not match.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")

        for emb, meta in zip(embeddings, metadata):
            emb = np.array(emb)
            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")
            self.metadata.append(meta)
            # Add the embedding to the FAISS index.
            self.index.add(np.expand_dims(emb.astype(np.float32), axis=0))

    def get_metadata(self, index):
        """
        Retrieves the metadata associated with a particular embedding index.
        
        Parameters:
            index (int): The index of the embedding.
        
        Returns:
            The metadata associated with the embedding.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.metadata):
            raise IndexError("Index out of bounds.")
        return self.metadata[index]

    def save(self, filepath):
        """
        Saves the current FaissBruteForce instance to a file.
        
        Parameters:
            filepath (str): The path to the file where the instance should be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissBruteForce instance from a file.
        
        Parameters:
            filepath (str): The path to the file from which to load the instance.
        
        Returns:
            An instance of FaissBruteForce loaded from the file.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


if __name__ == "__main__":
    # Example usage of FaissBruteForce class.

    # Initialize a FaissBruteForce index with embedding dimension 4.
    index = FaissBruteForce(dim=4)

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
    distances, indices, meta_results = index.index.search(query, k), None, None
    # The FAISS search directly returns distances and indices.
    distances, indices = index.index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]
    
    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_bruteforce_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissBruteForce.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))




