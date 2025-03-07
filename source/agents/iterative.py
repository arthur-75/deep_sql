import ollama

class IterativePromptingAgent:
    def __init__(self, model_name: str):
        """
        Initialise l'agent de raffinement itératif.
        
        :param model_name: Nom du modèle LLM utilisé.
        """
        self.model_name = model_name

    def generate_python_function(self, instruction: str, query_template: str, execution_feedback: dict) -> str:
        """
        Améliore un template SQL en fonction des retours d'exécution.
        
        :param query_template: Template SQL initial.
        :param execution_feedback: Feedback après exécution SQL et Python.
        :return: Nouveau template SQL amélioré.
        """
        self.message = self.prepare_prompt(instruction, query_template, execution_feedback)

        print(f"***ITERATIVE PROMPT***\n\n {self.message[0]['content']}\n\n\n")

        response = self.call_llm(self.message)

        query = response.message.content
        print(f"***First Response***\n\n {query}\n\n\n")

        # TODO: Implémenter la logique LLM pour l'ajustement des requêtes
        return query_template
    

    def prepare_prompt(self, instruction: str, query_template: str, execution_feedback: dict) -> str:
        """
        Prepare a structured prompt for an LLM agent to generate SQL queries.

        :param instruction: Instruction for the LLM (e.g., "Generate a SQL query").
        :param state: A list of previously generated SQL queries.
        :param error_history: A list of past errors encountered in SQL generation.
        :param table_description: The description of the database table for context.
        :return: A formatted prompt string ready for LLM input.
        """
        prompt = f"{instruction}\n\n"
        if len(execution_feedback)>0:
            prompt += f"{execution_feedback}\n\n"

        prompt += f"{query_template}\n\n"


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