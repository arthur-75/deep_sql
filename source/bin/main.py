import os
import sys

from source.utils.logger import setup_logger
from source.agents.curriculum import CurriculumAgent
from source.agents.iterative import IterativePromptingAgent
from source.executors.sql_executor import SQLExecutor
from source.executors.python_executor import execute_python_code
from source.library.storage import SQLLibrary
from source.library.retrieval import retrieve_similar_queries
import numpy as np

from source.utils.args import  ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser

# Initialisation du logger
logger = setup_logger()



def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialisation de la bibliothèque SQL
    sql_library = SQLLibrary(data_args.library_path)

    # Initialisation des agents LLM
    curriculum_agent = CurriculumAgent(model_name=model_args.curriculum_model, library=sql_library)
    iterative_agent = IterativePromptingAgent(model_name=model_args.iterative_model)

    # Initialisation des exécutants
    sql_executor = SQLExecutor(db_path=data_args.dataset_path)

    # Boucle principale d'exploration des requêtes SQL
    num_iterations =training_args.num_iterations
    
    for i in range(num_iterations):
        logger.info(f"🔄 Iteration {i+1}/{num_iterations}")

        # Étape 1: Génération d'une nouvelle requête SQL avec Curriculum Learning
        instruction = "Générer une requête SQL plus complexe en suivant le curriculum."
        state = sql_library.get_queries()
        error_history = []  # Historique des erreurs des requêtes précédentes
        new_sql_template = curriculum_agent.generate_query_template(instruction, state, error_history)

        logger.info(f"✅ Requête SQL générée : {new_sql_template}")

        # Étape 2: Exécution de la requête SQL
        sql_execution_result = sql_executor.execute_query(new_sql_template)

        if not sql_execution_result["success"]:
            logger.warning(f"❌ Erreur SQL détectée : {sql_execution_result['error']}")
            continue

        # Étape 3: Exécution de la fonction Python associée (si besoin)
        python_code = f"# Simule une transformation de requête SQL\nprint('{new_sql_template}')"
        python_execution_result = execute_python_code(python_code)

        if not python_execution_result["success"]:
            logger.warning(f"❌ Erreur d'exécution Python : {python_execution_result['error']}")
            continue

        # Étape 4: Analyse des feedbacks
        is_valid = analyze_feedback(sql_execution_result, python_execution_result)

        if not is_valid:
            logger.info("🔁 Raffinement de la requête SQL avec l'agent itératif.")
            new_sql_template = iterative_agent.refine_query(new_sql_template, sql_execution_result)
            continue

        # Étape 5: Vérification de la similarité avec les requêtes existantes
        query_vector = np.random.rand(768)  # Simule l'encodage de la requête en vecteur
        library_vectors = [np.random.rand(768) for _ in sql_library.get_queries().keys()]

        similar_queries = retrieve_similar_queries(query_vector, library_vectors)

        if similar_queries:
            logger.info(f"⚠️ La requête est trop similaire à d'autres. On passe à la suivante.")
            continue

        # Étape 6: Stockage de la requête SQL validée
        sql_library.add_query(new_sql_template, "Python function placeholder")
        logger.info(f"✅ Requête stockée avec succès !")

    logger.info("🎉 Fin du processus SQLExplore.")

if __name__ == "__main__":
    main()