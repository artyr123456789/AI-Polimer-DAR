def suggest_substitutions(predicted_material, similarity_index, compatibility_db, top_n=5):
    """
    Suggest substitute materials for a given predicted material based on similarity and compatibility.

    Args:
        predicted_material (str or int): The ID or name of the predicted material.
        similarity_index (SimilarityIndex): An object for querying similar materials.
        compatibility_db (CompatibilityDatabase): An object for checking material compatibility.
        top_n (int): Number of top substitutable options to return.

    Returns:
        list: A list of compatible substitute material IDs or names.
    """
    # Step 1: Query the similarity index for top similar materials
    similar_ids = similarity_index.query(predicted_material, k=10)  # Get top 10 similar materials

    # Step 2: Filter by compatibility using the compatibility database
    compatible_substitutes = [
        mat_id for mat_id in similar_ids
        if compatibility_db.is_compatible(predicted_material, mat_id)
    ]

    # Step 3: Return the top N compatible substitutes
    return compatible_substitutes[:top_n]