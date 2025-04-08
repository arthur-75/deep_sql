import requests
from bs4 import BeautifulSoup
from smolagents import tool,Tool
import sqlite3
from typing import Dict, List, Tuple, Any, Union, Optional



def get_table(db_path):
    conn = connect_to_database(db_path)
    tables_info = get_tables_info(conn)
    table_samples = get_random_table_samples(conn)

    return conn,tables_info,table_samples

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


# Setup database connection
def connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database and return connection object.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    return conn



def get_tables_info(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get all table names and their schemas from the database in a readable format.
    
    Args:
        conn: SQLite database connection
    
    Returns:
        Dictionary containing tables and their formatted schemas
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    schemas = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        table_info = cursor.fetchall()
        # Format the schema in a more readable way
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

def get_random_table_samples(conn: sqlite3.Connection, limit: int = 10) -> Dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    samples = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {limit};")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            samples[table] = [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            samples[table] = {"error": str(e)}
    
    return samples

def get_table_samples(conn: sqlite3.Connection, limit: int = 5) -> Dict[str, Any]:
    """
    Get sample rows from each table in the database in a readable format.
    
    Args:
        conn: SQLite database connection
        limit: Maximum number of rows to fetch per table
    
    Returns:
        Dictionary with formatted table samples
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
            
            # Format the sample data as readable rows
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

