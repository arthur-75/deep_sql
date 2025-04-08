import requests
from bs4 import BeautifulSoup
from smolagents import tool,Tool
import sqlite3
from typing import Dict, List, Tuple, Any, Union, Optional




@tool
def get_synonym(word: str) -> str:
    """
    Retrieves synonyms online for a given word from an API.
    The agent can use this tool get inspierd to use new term. 

    Args:
        word (str): The word for which synonyms are needed.

    Returns:
        list: list of synonyms, or an error message if none are found.
    """
    try:
        url = f'https://www.thesaurus.com/browse/{word}'
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return f"Error: Unable to fetch data (status code {response.status_code})."

        soup = BeautifulSoup(response.text, 'lxml')
        section = soup.find('section', {'data-type': 'synonym-antonym-module'})

        if not section:
            return "No synonyms found."

        strong_match = section.find("ul").find_all("li")

        if not strong_match:
            return "No synonyms available."

        synonyms = [i.text.strip() for i in strong_match]
        return synonyms

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to connect to Thesaurus.com ({str(e)})."
    except Exception as e:
        return f"Unexpected error: {str(e)}."
get_synonym("here")





class ExecuteSQLTool(Tool):
    name = "execute_sql"
    description = (
        "Executes an SQL query (SQLite) to check if it's valid and returns results. "
        "SQL query input must be a string"
        #"If an error occurs, the error message is returned as a string."
    )

    inputs = {
        "sql_query": {
            "type": "string",
            "description": ("The string SQL query code to execute.")
        }
    }
    output_type = "string"  # Could be "object" or "string" depending on your framework

    def __init__(self, conn: sqlite3.Connection, **kwargs):
        """
        Args:
            conn: A live sqlite3.Connection to the target database.
        """
        super().__init__(**kwargs)  # Optional, for compatibility with base class
        self.conn = conn


    def forward(
            self, 
            sql_query: str
            ) -> str :
        """
        Executes the provided SQL query and returns the results.

        Args:
            sql_query (str): The SQL query to run.

        Returns:
            List[Tuple]: The raw results of the query. If an error occurs,
                         returns a single-element list with an error message tuple.
        """
        assert isinstance(sql_query, str), "Your SQL query must be a string."
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(sql_query)
            answ= cursor.fetchall()
            print(str(answ))
            
            if len(answ)==0:raise ValueError(f"Wrong, no results it returns an empty list")
            return sql_query
        except Exception as e:
            return f"Error executing SQL: {str(e)} for the query: {sql_query}"

