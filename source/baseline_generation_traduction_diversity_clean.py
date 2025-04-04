import sqlite3
import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from smolagents import CodeAgent, tool, HfApiModel, OpenAIServerModel
from openai import OpenAI
from squall_Table import get_table_sale

# Initialisation du client OpenAI et configuration du modèle
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Choisir le modèle en fonction de la disponibilité des clés API
if os.environ.get("OPENAI_API_KEY"):
    model = OpenAIServerModel("gpt-4o")
else:
    model = HfApiModel()

# ---------------------- SERVICES D'EMBEDDING ----------------------

class EmbeddingService:
    """Service pour générer et comparer des embeddings vectoriels."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def generate(self, text: str) -> List[float]:
        """Génère un embedding pour un texte donné."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding: {e}")
            # Retourner un vecteur vide en cas d'erreur
            return [0.0] * 1536  # Dimension par défaut de text-embedding-3-small
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux embeddings."""
        import numpy as np
        # Convertir en arrays numpy pour le calcul
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calcul de la similarité cosinus
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# Instanciation du service d'embeddings
embedding_service = EmbeddingService()

# ---------------------- FONCTIONS UTILITAIRES ----------------------

def call_gpt(prompt: str, model_name: str = "gpt-4o", temperature: float = 0) -> str:
    """
    Appelle l'API OpenAI pour obtenir une réponse.
    
    Args:
        prompt: Le texte à envoyer au modèle
        model_name: Le modèle à utiliser ("gpt-3.5-turbo", "gpt-4" ou "gpt-4o")
        temperature: Paramètre de température pour la génération
        
    Returns:
        La réponse du modèle
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        wait_time = 2
        print(f"Erreur lors de l'appel à GPT: {e}")
        print(f"Attente de {wait_time} secondes avant de réessayer...")
        time.sleep(wait_time)
        return call_gpt(prompt, model_name, temperature)

# ---------------------- OUTILS POUR LES AGENTS ----------------------

@tool
def connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connecte à la base de données SQLite et retourne l'objet de connexion.
    
    Args:
        db_path: Chemin vers le fichier de base de données SQLite
    
    Returns:
        Objet de connexion SQLite
    """
    conn = sqlite3.connect(db_path)
    return conn

@tool
def get_tables_info(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Récupère les noms de toutes les tables et leurs schémas dans un format lisible.
    
    Args:
        conn: Connexion à la base de données SQLite
    
    Returns:
        Dictionnaire contenant les tables et leurs schémas formatés
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    schemas = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        table_info = cursor.fetchall()
        # Format le schéma de manière plus lisible
        formatted_columns = []
        for col in table_info:
            col_id, name, type_, notnull, default, pk = col
            attributes = []
            if pk:
                attributes.append("PRIMARY KEY")
            if notnull:
                attributes.append("NOT NULL")
            if default is not None:
                attributes.append(f"DEFAULT {default}")
            
            attr_str = ", ".join(attributes)
            if attr_str:
                formatted_columns.append(f"{name} ({type_}) - {attr_str}")
            else:
                formatted_columns.append(f"{name} ({type_})")
                
        schemas[table] = formatted_columns
    
    return {"tables": tables, "schemas": schemas}

@tool
def get_table_samples(conn: sqlite3.Connection, limit: int = 5) -> Dict[str, Any]:
    """
    Récupère des échantillons de lignes de chaque table dans un format lisible.
    
    Args:
        conn: Connexion à la base de données SQLite
        limit: Nombre maximum de lignes à récupérer par table
    
    Returns:
        Dictionnaire avec des échantillons de tables formatés
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    samples = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT {limit};")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Formate les échantillons de données en lignes lisibles
            formatted_rows = []
            for row in rows:
                formatted_row = {}
                for i, value in enumerate(row):
                    formatted_row[columns[i]] = value
                formatted_rows.append(formatted_row)
                
            samples[table] = formatted_rows
        except Exception as e:
            samples[table] = {"error": str(e)}
    
    return samples

@tool
def execute_sql(conn: sqlite3.Connection, sql_query: str) -> List[Tuple]:
    """
    Exécute une requête SQL et retourne les résultats.
    
    Args:
        conn: Connexion à la base de données SQLite
        sql_query: Requête SQL à exécuter
    
    Returns:
        Liste de tuples résultant de la requête
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        return cursor.fetchall()
    except Exception as e:
        return [(f"Erreur lors de l'exécution SQL: {str(e)}",)]

@tool
def generate_text_embedding(text: str) -> List[float]:
    """
    Génère un embedding vectoriel pour un texte donné.
    
    Args:
        text: Le texte à convertir en vecteur
        
    Returns:
        Liste de valeurs représentant l'embedding
    """
    return embedding_service.generate(text)

@tool
def compare_embeddings(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calcule la similarité cosinus entre deux embeddings.
    
    Args:
        embedding1: Premier embedding
        embedding2: Deuxième embedding
        
    Returns:
        Score de similarité entre 0 et 1
    """
    return embedding_service.calculate_similarity(embedding1, embedding2)

@tool
def check_question_similarity(library: List[Dict[str, Any]], new_question: str, threshold: float = 0.85) -> Tuple[bool, str]:
    """
    Vérifie si une nouvelle question est trop similaire aux questions existantes
    en utilisant les embeddings.
    
    Args:
        library: Liste des entrées existantes
        new_question: Nouvelle question à vérifier
        threshold: Seuil de similarité au-delà duquel la question est considérée comme dupliquée
        
    Returns:
        Tuple (est_unique, message)
    """
    if not library:
        return (True, "Première question - automatiquement acceptée")
    
    # Générer l'embedding de la nouvelle question
    new_embedding = generate_text_embedding(new_question)
    
    # Comparer avec les questions existantes
    similarities = []
    for entry in library:
        # Si l'embedding n'existe pas encore, le générer et l'ajouter
        if "question_embedding" not in entry:
            entry["question_embedding"] = generate_text_embedding(entry["question"])
        
        # Calculer la similarité cosinus
        similarity = compare_embeddings(new_embedding, entry["question_embedding"])
        similarities.append(similarity)
    
    max_similarity = max(similarities)
    
    if max_similarity > threshold:
        return (False, f"Question trop similaire à une question existante (similarité: {max_similarity:.2f})")
    
    return (True, f"Question acceptée (similarité maximale: {max_similarity:.2f})")

@tool
def validate_sql_query(conn: sqlite3.Connection, sql_query: str) -> Dict[str, Any]:
    """
    Valide qu'une requête SQL s'exécute correctement et retourne des résultats.
    
    Args:
        conn: Connexion à la base de données SQLite
        sql_query: Requête SQL à valider
    
    Returns:
        Dictionnaire avec le statut de validation et les résultats ou message d'erreur
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Vérifie si nous avons obtenu des résultats
        if not results or len(results) == 0:
            return {
                "valid": False,
                "error": "La requête s'est exécutée avec succès mais n'a retourné aucun résultat",
                "results": []
            }
            
        # Récupère les noms de colonnes pour une meilleure lisibilité
        column_names = [description[0] for description in cursor.description]
        
        # Formate les résultats en liste de dictionnaires
        formatted_results = []
        for row in results:
            formatted_row = {}
            for i, value in enumerate(row):
                formatted_row[column_names[i]] = value
            formatted_results.append(formatted_row)
        
        return {
            "valid": True,
            "sql": sql_query,
            "results": formatted_results,
            "row_count": len(results),
            "column_names": column_names
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "results": []
        }

@tool
def check_translation_correctness(question: str, sql_query: str) -> Dict[str, Any]:
    """
    Checks whether the translation of the SQL query in question matches the initial question.
    
    Args:
    question: The initial question
    sql_query: The SQL query to be checked
        
    Returns:
    Dictionary with validation status and a message
    """
    prompt = f"""
        You are an expert in SQL and natural language processing.
        You need to check whether the SQL query corresponds correctly to the question asked.
        
        Question: {question}
    SQL query: {sql_query}
        
    Check whether the SQL query answers the question correctly and simply return "True" 
    if it does or "False" if it does not. Do not give an explanation.
    """
    
    result = call_gpt(prompt)
    
    if result.strip().lower() not in ["true", "false"]:
        clarification_prompt = f"""
        Your previous answer was: ‘{result}’.
        Simply answer with "True" if the SQL query is correct or "False" if it is incorrect.
        """
        result = call_gpt(clarification_prompt)
    
    is_valid = result.strip().lower() == "true"
    
    return {
        "valid": is_valid,
        "message": "The SQL query matches the question" if is_valid else "The SQL query does not match the question"
    }

# ---------------------- GESTION DE LA BIBLIOTHÈQUE ----------------------

def init_library(library_path: str = "sql_dataset_library.json") -> List[Dict[str, Any]]:
    """Initialise ou charge la bibliothèque existante"""
    if os.path.exists(library_path):
        with open(library_path, 'r') as f:
            return json.load(f)
    return []

def save_library(library: List[Dict[str, Any]], library_path: str = "sql_dataset_library.json"):
    """Sauvegarde la bibliothèque actuelle sur disque"""
    with open(library_path, 'w') as f:
        json.dump(library, f, indent=2)

# ---------------------- CRÉATION DES AGENTS ----------------------

def create_agents(model):
    """Crée les agents du pipeline avec accès aux outils appropriés"""
    
    question_generator = CodeAgent(
        model=model,
        tools=[generate_text_embedding, compare_embeddings, check_question_similarity],
        name="question_generator",
        description="Generates a new database question from the schema and sample data"
    )

    sql_translator = CodeAgent(
        model=model,
        tools=[validate_sql_query],
        name="sql_translator", 
        description="Translates natural language questions into SQL queries and return a SQL query"
    )

    validator = CodeAgent(
        model=model,
        tools=[check_translation_correctness],
        name="validator",
        description="Checks whether the SQL query corresponds correctly to the initial question"
    )
    
    return question_generator, sql_translator, validator

# ---------------------- FONCTION PRINCIPALE DE GÉNÉRATION ----------------------

def generate_question_and_sql(db_path: str, db_sale: Dict, library: List[Dict[str, Any]], max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Generates a question, its SQL query and checks its validity for a given table.
    
    Args:
    db_path: Path to the SQLite database file
    db_sale: Data from the 'sale' table (in text format)
    library: List of existing examples to check similarity
    max_retries: Maximum number of attempts to generate a single question
            
    Returns:
    Dictionary containing the question, SQL query and associated metadata.
    """
    # Connexion à la base de données
    conn = connect_to_database(db_path)
    
    # Récupérer les informations de la table
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    if not tables:
        print(f"Erreur: Aucune table trouvée dans {db_path}")
        return None
    
    table_name = tables[0]
    
    # Récupérer le schéma de la table
    cursor.execute(f"PRAGMA table_info({table_name});")
    table_info = cursor.fetchall()
    
    formatted_columns = []
    for col in table_info:
        col_id, name, type_, notnull, default, pk = col
        attributes = []
        if pk:
            attributes.append("PRIMARY KEY")
        if notnull:
            attributes.append("NOT NULL")
        if default is not None:
            attributes.append(f"DEFAULT {default}")
        
        attr_str = ", ".join(attributes)
        if attr_str:
            formatted_columns.append(f"{name} ({type_}) - {attr_str}")
        else:
            formatted_columns.append(f"{name} ({type_})")
    
    # Récupérer des échantillons de données
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    formatted_rows = []
    for row in rows:
        formatted_row = {}
        for i, value in enumerate(row):
            formatted_row[columns[i]] = value
        formatted_rows.append(formatted_row)
    
    # Créer les agents
    question_generator, sql_translator, validator = create_agents(model)
    
    sql_translator.python_executor.state["conn"] = conn # Passer la connexion à l'agent SQL

    # Boucle de génération avec vérification de similarité
    for retry_count in range(max_retries):
        # 1. Générer la question
        question_prompt = f"""
        You are a SQL question generator for a database with the following structure:
        
        TABLE: {table_name}
        
        TABLE SCHEME:
        {json.dumps(db_sale, indent=2)}
                
        From the table schema and sample data above, generate a clear and specific question
        that can be answered using SQL on this table.
                
                IMPORTANT: Make sure your question:
        1. Is specific enough to be translated into SQL
        2. Has an answer in the database (based on samples)
        3. Requires only one SQL query
        4. Is written in plain language
        5. Uses classic agent output
        """
        
        question = question_generator.run(question_prompt)
        print(f"Question générée: {question}")
        
        # Vérifier si la question est suffisamment originale
        print("Vérification de la similarité de la question...")
        is_unique, message = check_question_similarity(library, question)
        print(message)
        
        if not is_unique:
            print(f"The question is too similar to existing questions. Attempt {retry_count+1}/{max_retries}...")
            if retry_count < max_retries - 1:
                continue
            else:
                print("Maximum number of attempts reached, last question generated used.")
    
        # 2. Traduire en SQL
        sql_prompt = f"""
        You are an SQL expert translating natural language questions into SQL.
        
        TABLE: {table_name}
        
            TABLE DIAGRAM:
        """
        
        for column in formatted_columns:
            sql_prompt += f"  - {column}\n"
        
        sql_prompt += "\Some datas:\n"
        for i, row in enumerate(formatted_rows):
            sql_prompt += f"  {row}\n"
        
        sql_prompt += f"""
        Question: {question}
        
        Write a valid SQL query that answers this question correctly and return the SQL.
        Ensure that the SQL query will return results based on the sample data shown.
        You have access to "conn" object. Just use conn = conn 
        final_answer the sql query.
        """
        
        sql_query = sql_translator.run(sql_prompt)
        print(f"SQL generated: {sql_query}")
        
        # 3. Exécuter la requête pour vérifier qu'elle fonctionne
        try:
            max_retries_sql = 3
            retry_count_sql = 0
            result = execute_sql(conn, sql_query)
            print(f"request result: {result}")
            
            # Vérifier que la requête retourne des résultats
            if not result or (len(result) == 1 and str(result[0][0]).startswith("Erreur")):
                print("The query did not return any valid results, try again...")
                
                # add 3 attempts to generate a new SQL query
                retry_count_sql += 1
                while retry_count_sql < max_retries_sql:
                        
                    sql_query = sql_translator.run(sql_prompt)
                    print(f"SQL generated: {sql_query}")
                    # Vérifier que la requête a retourné des résultats
                    result = execute_sql(conn, sql_query)
                    if not result and (len(result) == 1 and str(result[0][0]).startswith("Erreur")):
                        print("The query did not return any valid results, try again...")
                        retry_count_sql += 1
                
                    else:
                        # 4. Valider la correspondance entre la question et la requête SQL
                        validation_result = validator.run(f"""
                        Check whether this SQL query answers the question:
                                
                        Question: {question}
                        SQL query: {sql_query}
                        Answer: {result}
                                
                        Return True if the query answers the question, False otherwise based on the agent format output.
                        """)
                        
                        # Si la validation échoue, réessayer la traduction jusqu'à 3 fois
                        validation_attempts = 0
                        while str(validation_result) != "True" and validation_attempts < 3:
                            validation_attempts += 1
                            print(f"Validation failed, attempt to correct {validation_attempts}/3...")
                            
                            correction_prompt = f"""
                            The following SQL query does not answer the question correctly. Please correct it:
                                        
                            Question: {question}
                            Incorrect SQL query: {sql_query}
                                        
                            Return final_answer with the sql query, based on the agent format output.
                            
                            """
                            
                            sql_query = sql_translator.run(correction_prompt)
                            print(f"SQL corrigé: {sql_query}")
                            result = execute_sql(conn, sql_query)

                            
                            validation_result = validator.run(f"""
                            Check whether this SQL query answers the question:
                            
                            Question: {question}
                            SQL query: {sql_query}
                            Answer: {result}
                                        
                            Return True if the query answers the question, False otherwise. based on the agent format output
                            """)
                        
                        if str(validation_result) != "True":
                            print("Validation fails after several attempts, go to a new question...")
                            continue

                        # Tout est bon, générer les embeddings et retourner le résultat
                        question_embedding = generate_text_embedding(question)
                        sql_embedding = generate_text_embedding(sql_query)
                        return {
                                "question": question,
                                "sql": sql_query,
                                "result": result,
                                "valid": True,
                                "question_embedding": question_embedding,
                                "sql_embedding": sql_embedding
                            }
                if not result or (len(result) == 1 and result[0][0].startswith("Erreur")):
                    print("The query did not return any valid results after several attempts, go to a new question...")
                    continue
                
            
        except Exception as e:
            print(f"Error during SQL execution: {str(e)}")
            continue

        
    
    # Si nous arrivons ici, c'est que nous n'avons pas réussi à générer une question/requête valide
    print("Generation fails after several attempts")
    return None

# ---------------------- FONCTION PRINCIPALE ----------------------

def main():
    """Fonction principale pour exécuter le pipeline de génération"""
    # Charger ou initialiser la bibliothèque
    library = init_library()
    library_with_embeddings = init_library("sql_dataset_with_embeddings.json")
    
    # Récupérer les tables
    squall_table_id_by_id, wtq_table_by_id, common_ids = get_table_sale()
    
    # Générer des questions pour les 100 premières tables
    for i in tqdm(range(10)):
        table_id = common_ids[i]
        table = wtq_table_by_id[table_id]
        table_db_id = squall_table_id_by_id[table_id]
        
        print(f"\nTraitement de la table {i+1}/100 (ID: {table_id})")
        
        # Générer une question, une requête SQL et vérifier leur validité
        entry = generate_question_and_sql(
            os.path.join("../data/tables/db", f"{table_db_id}.db"),
            table,
            library_with_embeddings
        )
        
        if entry is None:
            print(f"Échec pour la table {table_id}, passage à la suivante")
            continue
        
        # Ajouter l'ID de la table
        entry["table_id"] = table_id
        
        # Ajouter à la bibliothèque avec embeddings
        library_with_embeddings.append(entry)
        
        # Créer une version simplifiée sans embeddings
        simplified_entry = {
            "question": entry["question"],
            "sql": entry["sql"],
            "result": entry["result"],
            "table_id": table_id
        }
        
        library.append(simplified_entry)
        
        # Sauvegarder régulièrement
        if (i + 1) % 1 == 0 :
            print(f"Sauvegarde intermédiaire après {i+1} tables")
            save_library(library)
            save_library(library_with_embeddings, "sql_dataset_with_embeddings.json")
    
    # Sauvegarde finale
    save_library(library)
    save_library(library_with_embeddings, "sql_dataset_with_embeddings.json")
    
    print("Génération terminée avec succès!")

if __name__ == "__main__":
    main()