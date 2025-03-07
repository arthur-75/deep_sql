import ollama
from source.library.utils import extract_sql_from_text

class CurriculumAgent:
    def __init__(self, model_name: str, library: object):
        """
        Initialise l'agent de curriculum learning.
        
        :param model_name: Nom du modèle LLM utilisé (ex: "llama").
        :param library: Instance de la bibliothèque stockant les requêtes SQL.
        """
        self.model_name = model_name
        self.library = library
        self.message = None

    def generate_query_template(self, instruction: str, state: dict, error_history: list, table_description: str) -> str:
        """
        Génère un nouveau template SQL basé sur l'état actuel du curriculum.
        
        :param instruction: Instruction textuelle pour guider la génération.
        :param state: État actuel des templates SQL générés.
        :param error_history: Historique des erreurs de génération.
        :return: Nouveau template SQL sous forme de chaîne de caractères.
        """
        # TODO: Implémenter la logique avec LLM
        if not self.message:
            self.message = self.prepare_prompt(instruction, state, error_history, table_description)

        print(f"***CURRICULUM PROMPT***\n\n {self.message[0]['content']}\n\n\n")

        response = self.call_llm(self.message)

        print(f"***First Response***\n\n {response.message.content}\n\n\n")


        query = extract_sql_from_text(response.message.content)
            
        return query




    def prepare_prompt(self, instruction: str, state: list, error_history: list, table_description: str) -> str:
        """
        Prepare a structured prompt for an LLM agent to generate SQL queries.

        :param instruction: Instruction for the LLM (e.g., "Generate a SQL query").
        :param state: A list of previously generated SQL queries.
        :param error_history: A list of past errors encountered in SQL generation.
        :param table_description: The description of the database table for context.
        :return: A formatted prompt string ready for LLM input.
        """
        prompt = "### SQL Query Generation Task ###\n"
        prompt += f"📝 **Instruction:** {instruction}\n\n"

        prompt += "📌 **Table Description:**\n"
        prompt += table_description + "\n\n"

        if state:
            prompt += "📂 **Previous SQL Queries:**\n"
            for idx, sql_query in enumerate(state, 1):
                prompt += f"{idx}. {sql_query}\n"
            prompt += "\n"

        if error_history:
            prompt += "⚠️ **Error History:**\n"
            for idx, error in enumerate(error_history, 1):
                prompt += f"{idx}. {error}\n"
            prompt += "\n"

        prompt += "💡 **Objective:**\n"
        prompt += "Generate a new SQL query based on the given table schema while avoiding previous mistakes.\n"


        prompt =[
                {
                    'role': 'user',
                    'content': prompt,
                },
                ]
        
        return prompt
    


    def call_llm(self, messages: list) -> str:
        """
        Appelle le modèle LLM pour générer un nouveau template SQL.
        
        :param messages: Liste des messages d'entrée pour le modèle.
        :return: Template SQL généré par le modèle.
        """


        
        response = ollama.chat(
                model=self.model_name,
                messages=messages)
                

        return response

    