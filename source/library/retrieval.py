from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_similar_queries(new_query_vector: np.ndarray, library_vectors: list, threshold: float = 0.8) -> list:
    """
    Récupère les requêtes SQL similaires dans la bibliothèque en fonction de la similarité cosinus.

    :param new_query_vector: Vecteur encodé de la requête SQL.
    :param library_vectors: Liste des vecteurs de la bibliothèque SQL.
    :param threshold: Seuil de similarité cosinus pour la récupération.
    :return: Liste des requêtes similaires.
    """
    similarities = cosine_similarity([new_query_vector], library_vectors)
    similar_queries = [idx for idx, sim in enumerate(similarities[0]) if sim >= threshold]
    return similar_queries

    