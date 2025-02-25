import pandas as pd
from material_embedder import MaterialEmbedder  # Предполагается, что MaterialEmbedder определен в отдельном модуле

def find_substitutes(query_material, db, k=10, threshold=0.8, compatibility_file="data/element_compatibility.csv"):
    """
    Find substitute materials for a given query material based on similarity and compatibility.

    Args:
        query_material (str): Name of the material to find substitutes for.
        db (SimilarityIndex): FAISS-based similarity index containing material embeddings.
        k (int): Number of nearest neighbors to retrieve.
        threshold (float): Minimum compatibility score for a material to be considered a substitute.
        compatibility_file (str): Path to the CSV file containing compatibility scores.

    Returns:
        list: List of substitute materials that meet the compatibility threshold.
    """
    # Step 1: Get embedding for the query material
    embedder = MaterialEmbedder()
    embedding = embedder.get_embedding(query_material)

    # Step 2: Query the similarity index
    distances, indices = db.query(embedding, k)

    # Step 3: Load compatibility data
    compatibility_df = pd.read_csv(compatibility_file)
    
    # Create a dictionary for quick lookup of compatibility scores
    compatibility_dict = {}
    for _, row in compatibility_df.iterrows():
        material_a, material_b, score = row['Material A'], row['Material B'], row['Compatibility Score']
        compatibility_dict[(material_a, material_b)] = score
        compatibility_dict[(material_b, material_a)] = score  # Ensure symmetry

    # Step 4: Retrieve candidate materials from the database
    candidate_materials = [db.embeddings[i] for i in indices]  # Assuming db.embeddings contains material names

    # Step 5: Filter candidates by compatibility
    substitutes = []
    for material in candidate_materials:
        # Check compatibility score between query_material and candidate material
        compatibility_key = (query_material, material)
        compatibility_score = compatibility_dict.get(compatibility_key, 0.0)  # Default to 0 if not found
        if compatibility_score >= threshold:
            substitutes.append(material)

    return substitutes