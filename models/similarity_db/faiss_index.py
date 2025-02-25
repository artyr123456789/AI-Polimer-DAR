import faiss
import numpy as np

class SimilarityIndex:
    def __init__(self, dimension=768, use_cosine_similarity=False):
        """
        Initialize the similarity index.

        Args:
            dimension (int): Dimensionality of the embedding vectors.
            use_cosine_similarity (bool): If True, normalize vectors for cosine similarity.
        """
        self.dimension = dimension
        self.use_cosine_similarity = use_cosine_similarity

        # Choose an appropriate FAISS index
        if use_cosine_similarity:
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance for Euclidean similarity

        # Optional: Train the index if using approximate methods like IVF or HNSW
        # For simplicity, we use exact search here
        self.embeddings = []  # Store embeddings for reference

    def add_materials(self, embeddings):
        """
        Add material embeddings to the index.

        Args:
            embeddings (list of numpy.ndarray): List of embedding vectors.
        """
        if not isinstance(embeddings, list) or not all(isinstance(vec, np.ndarray) for vec in embeddings):
            raise ValueError("Embeddings must be a list of numpy arrays.")

        if any(vec.shape[0] != self.dimension for vec in embeddings):
            raise ValueError(f"All embeddings must have dimension {self.dimension}.")

        # Normalize vectors if using cosine similarity
        if self.use_cosine_similarity:
            embeddings = [vec / np.linalg.norm(vec) for vec in embeddings]

        # Convert to float32 and add to the index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.embeddings.extend(embeddings)

    def query(self, embedding, k=5):
        """
        Query the index for the k most similar materials.

        Args:
            embedding (numpy.ndarray): Query embedding vector.
            k (int): Number of nearest neighbors to return.

        Returns:
            tuple: (distances, indices) where distances are similarity scores and indices are the indices of the materials.
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding must have dimension {self.dimension}.")

        # Normalize query vector if using cosine similarity
        if self.use_cosine_similarity:
            embedding = embedding / np.linalg.norm(embedding)

        # Search the index
        distances, indices = self.index.search(np.array([embedding]).astype('float32'), k)

        # Return results
        return distances[0], indices[0]

    def clear_index(self):
        """
        Clear all embeddings from the index.
        """
        self.index.reset()
        self.embeddings.clear()