class SQLLibrary:
    def __init__(self):
        """
        Initialise la bibliothèque SQL pour stocker les requêtes et templates.
        """
        self.storage = {}

    def add_query(self, query: str, function: str) -> None:
        """
        Ajoute une requête SQL et sa fonction associée dans la bibliothèque.
        
        :param query: Template SQL généré.
        :param function: Fonction Python associée à la requête.
        """
        self.storage[query] = function

    def get_queries(self) -> dict:
        """
        Retourne toutes les requêtes stockées.
        
        :return: Dictionnaire des requêtes et fonctions associées.
        """
        return self.storage