#### BAseline file
"""
Partir d'un table sale pour générer une question et une réponse basé sur la table
"""

import os
from openai import OpenAI
from langchain.prompts import PromptTemplate
from squall_Table import get_table_sale
import time
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

def call_gpt_gpt(type, prompt_utilisateur, template=None, only_cast_to_template=False, temperature=0):
    """
    Call OpenAI's GPT models with different configurations.
    Falls back to Gemini if OpenAI's content filter is triggered.

    Args:
        type (str): Model type ('gpt-3.5-turbo', 'gpt-4', or 'gpt-4o')
        prompt_utilisateur: Either a string prompt or a PromptTemplate
        template (str, optional): Template string for formatting the prompt
        only_cast_to_template (bool): Whether to only format the template
        temperature (float): Temperature parameter for response randomness

    Returns:
        str: The model's response text
    """
    try:
        if type == "gpt3":
            model = "gpt-3.5-turbo"
        elif type == "gpt4":
            model = "gpt-4"
        else:
            model = "gpt-4o"

        if isinstance(prompt_utilisateur, PromptTemplate):
            prompt_text = prompt_utilisateur.format({})
        elif template is None and isinstance(prompt_utilisateur, str):
            prompt_text = prompt_utilisateur
        else:
            prompt_text = template.format(prompt=prompt_utilisateur)

        messages = [{"role": "user", "content": prompt_text}]
        response = client.chat.completions.create(model=model,
            messages=messages,
            temperature=temperature)
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        
        wait_time = 5
        print(f"Error: {e}")
        print(f"Waiting {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        return call_gpt_gpt(type, prompt_utilisateur, template, only_cast_to_template, temperature)

        



def generate_question(table, type="gpt-4o"):
    """
    Generate a question based on a table and a template.

    Args:
        table (dict): A dictionary representing the table
        template (str): A template string for the question
        type (str): Model type ('gpt-3.5-turbo', 'gpt-4', or 'gpt-4o')

    Returns:
        str: The generated question
    """
    prompt = """

    Given the following table, your job is to generate a question in natural language and it answer. 
    in this format without any other text:
    Question? answer 

    example: 

    What is the name of the first person in the table? John


    Table:
    {}
    """.format(table)
    
    return call_gpt_gpt(type, prompt)


# main section
def main():
    squall_table_id_by_id, wtq_table_by_id, common_ids = get_table_sale()
    
    save_question = []

    # for the 100 first tables, generate a question and answer.
    for i in range(100):
        table_id = common_ids[i]
        table = wtq_table_by_id[table_id]
        question = generate_question(table)
        #print(question)
        #split question by "?"
        question = question.split("?")
        save_question.append({"table_id": table_id, "question": str(question[0]+"?"), "answer": question[1]})
    # Save the questions to a file
    with open("questions.json", "w") as f:
        import json
        json.dump(save_question, f, indent=2)

if __name__ == "__main__":
    main()