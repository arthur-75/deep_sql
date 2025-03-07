import os
import sys

from source.utils.logger import setup_logger
from source.agents.curriculum import CurriculumAgent
from source.agents.iterative import IterativePromptingAgent
from source.executors.sql_executor import SQLExecutor
from source.executors.python_executor import execute_python_code
from source.library.storage import SQLLibrary
from source.library.tables import TableManager
import numpy as np

from source.utils.args import  ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser

import json


# Unset potential debug variables for Ollama
os.environ.pop("OLLAMA_DEBUG", None)
os.environ.pop("OLLAMA_LOG_LEVEL", None)
os.environ.pop("OLLAMA_VERBOSE", None)

# Initialisation du logger
logger = setup_logger()





def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialisation de la bibliothèque SQL
    sql_library = SQLLibrary(data_args, model_args)
    table_manager = TableManager(data_args.database_path)

    # Initialisation des agents LLM
    curriculum_agent = CurriculumAgent(model_name=model_args.curriculum_model)
    iterative_agent = IterativePromptingAgent(model_name=model_args.iterative_model)


    

    # Open and read the file
    with open(data_args.curriculum_instruction, "r", encoding="utf-8") as file:
        curriculum_instruction = file.read()




    for i in range(training_args.num_iterations):

        logger.info(f"\n"*11)
        logger.info(f"🔄 Iteration {i+1}/{training_args.num_iterations}")



        # Étape 1: Récupérer les informations de la table et l'état actuel de la bibliothèque
        state = sql_library.get_sql(random_=True, num_q=2)
        logger.info(f"Library State : {state}\n\n")
        error_history = []  # Historique des erreurs des requêtes précédentes
        table_description, table_path = table_manager.get_random_table_info()
        logger.info(f"table {table_path}, table_description {table_description}\n\n")


        # Étape 2: Générer la requête SQL
        new_sql_template = curriculum_agent.generate_query_template(curriculum_instruction, state, error_history, table_description)
        logger.info(f"✅ Requête SQL générée : {new_sql_template}\n\n")


        # Étape 3: vérifier l'execution SQL
        sql_executor = SQLExecutor(table_path)
        sql_execution_result = sql_executor.execute_query(new_sql_template)
        if not sql_execution_result["success"]:
            logger.warning(f"❌ Erreur SQL détectée : {sql_execution_result['error']}")
            continue

        # Étape 4: vérifier similarité si faill -> Etape 2 avec error_history

        # Étape 5: Vérifier la différence si faill -> Etape 2 avec error_history
        
        # Étape 6: Création de la fonction python
        python_code = f"print('hellow world')"

        # Étape 7: Execution de la fonction python

        # Étape 8: Stockage de la requête SQL validée
        sql_library.add_query(new_sql_template, python_func=python_code, save=True)
        logger.info(f"✅ Requête stockée avec succès !")


    logger.info("🎉 Fin du processus SQLExplore.")

if __name__ == "__main__":
    main()