import sqlite3

class SQLExecutor:
    def __init__(self, db_path: str):
        """
        Initialise l'exécuteur SQL avec une base de données spécifique.
        
        :param db_path: Chemin vers la base de données SQLite.
        """
        self.db_path = db_path

    def execute_query(self, query: str) -> dict:
        """
        Exécute une requête SQL et retourne les résultats.

        :param query: La requête SQL à exécuter.
        :return: Un dictionnaire contenant le statut et les résultats.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            conn.commit()
            conn.close()
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}