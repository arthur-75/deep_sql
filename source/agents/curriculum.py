import ollama

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

    def generate_query_template(self, instruction: str, state: dict, error_history: list) -> str:
        """
        Génère un nouveau template SQL basé sur l'état actuel du curriculum.
        
        :param instruction: Instruction textuelle pour guider la génération.
        :param state: État actuel des templates SQL générés.
        :param error_history: Historique des erreurs de génération.
        :return: Nouveau template SQL sous forme de chaîne de caractères.
        """
        # TODO: Implémenter la logique avec LLM
        if not self.message:
            self.message = self.prepare_prompt(instruction, state, error_history)

        
        query = self.call_llm(self.message)

            
        return query

    def prepare_prompt(self, instruction: str, state: dict, error_history: list, table_description: str) -> str:

        self.messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"**Existing Queries**: {state} \n**Table Description**: {table_description}"},
            ]

        if error_history is not None:
            # todo
            pass

    def call_llm(self, messages: list) -> str:
        """
        Appelle le modèle LLM pour générer un nouveau template SQL.
        
        :param messages: Liste des messages d'entrée pour le modèle.
        :return: Template SQL généré par le modèle.
        """

        
        response = ollama.chat(
                model=self.model_name,
                messages=messages)
                
        print(response)
        return response

    