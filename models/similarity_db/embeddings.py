from sentence_transformers import SentenceTransformer
import numpy as np

class MaterialEmbedder:
    def __init__(self):
        """
        Initialize the MaterialEmbedder with a pre-trained chemical-aware embedding model.
        Uses 'all-mpnet-base-v2' from SentenceTransformers.
        """
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Pre-trained chemical-aware embeddings
        self.cache = {}  # Cache for storing previously computed embeddings

    def get_embedding(self, material_names):
        """
        Get embeddings for one or more material names.

        Args:
            material_names (str or list of str): Material name(s) to embed.

        Returns:
            numpy.ndarray: Embedding(s) for the material(s).
        """
        # Handle single string input by converting it to a list
        if isinstance(material_names, str):
            material_names = [material_names]

        # Check cache and filter out materials that need embedding
        cached_embeddings = []
        uncached_materials = []
        for material in material_names:
            if material in self.cache:
                cached_embeddings.append(self.cache[material])
            else:
                uncached_materials.append(material)

        # Compute embeddings for uncached materials
        if uncached_materials:
            new_embeddings = self.model.encode(uncached_materials, convert_to_numpy=True)
            # Update cache
            for material, embedding in zip(uncached_materials, new_embeddings):
                self.cache[material] = embedding

        # Combine cached and new embeddings
        all_embeddings = cached_embeddings + list(new_embeddings)

        # Return as a single array if input was a single material, otherwise return as a list
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.array(all_embeddings)

    def clear_cache(self):
        """
        Clear the embedding cache.
        """
        self.cache.clear()