import os
import sys

from source.utils.logger import setup_logger
from source.agents.curriculum import CurriculumAgent
from source.agents.iterative import IterativePromptingAgent
from source.executors.sql_executor import SQLExecutor
from source.executors.python_executor import execute_python_code
from source.library.storage import SQLLibrary
from source.library.tables import TableManager
from source.library.retrieval import retrieve_similar_queries
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
    curriculum_agent = CurriculumAgent(model_name=model_args.curriculum_model, library=sql_library)
    iterative_agent = IterativePromptingAgent(model_name=model_args.iterative_model)

    # Chargement des prompts
    with open("../data/prompts.json", "r", encoding="utf-8") as json_file:
        prompts = json.load(json_file)
    

    
    for i in range(training_args.num_iterations):
        logger.info(f"🔄 Iteration {i+1}/{training_args.num_iterations}")


        state = sql_library.get_sql(random_=True, num_q=2)
        logger.info(f"Library State : {state}\n\n")


        error_history = []  # Historique des erreurs des requêtes précédentes

        table_description, table_path = table_manager.get_random_table_info()

        new_sql_template, sql_embd = curriculum_agent.generate_query_template(prompts["curriculum_instruction"], state, error_history, table_description)

        logger.info(f"✅ Requête SQL générée : {new_sql_template}\n\n")


        sql_executor = SQLExecutor(table_path)
        sql_execution_result = sql_executor.execute_query(new_sql_template)



        if not sql_execution_result["success"]:
            logger.warning(f"❌ Erreur SQL détectée : {sql_execution_result['error']}")
            continue


        # Étape 3: Exécution de la fonction Python associée (si besoin)
        python_code = f"print('hellow world')"
    

        # Étape 4: vérifier similarité

        # Étape 5: Vérifier la différence


        # Étape 6: Stockage de la requête SQL validée
        sql_library.add_query(new_sql_template, python_func=python_code,sql_embd=sql_embd, save=True)
        logger.info(f"✅ Requête stockée avec succès !")


    logger.info("🎉 Fin du processus SQLExplore.")

if __name__ == "__main__":
    main()