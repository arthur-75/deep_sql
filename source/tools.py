import requests
from bs4 import BeautifulSoup
from smolagents import tool
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
        str: A comma-separated list of synonyms, or an error message if none are found.
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

        synonyms = ", ".join([i.text.strip() for i in strong_match])
        return synonyms

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to connect to Thesaurus.com ({str(e)})."
    except Exception as e:
        return f"Unexpected error: {str(e)}."
get_synonym("here")


# Setup database connection
@tool
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



@tool
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


@tool
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

@tool
def execute_sql(conn: sqlite3.Connection, sql_query: str) -> List[Tuple]:
    """
    Execute SQL query and return results.
    
    Args:
        conn: SQLite database connection
        sql_query: SQL query to execute
    
    Returns:
        List of tuple results from the query
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        return cursor.fetchall()
    except Exception as e:
        return [(f"Error executing SQL: {str(e)}",)]
    

@tool
def validate_sql_query(conn: sqlite3.Connection, sql_query: str) -> Dict[str, Any]:
    """
    Validate that an SQL query runs correctly and returns results.
    
    Args:
        conn: SQLite database connection
        sql_query: SQL query to validate
    
    Returns:
        Dictionary with validation status and results or error message
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Check if we got any results
        if not results or len(results) == 0:
            return {
                "valid": False,
                "error": "Query executed successfully but returned no results",
                "results": []
            }
            
        # Get column names for better readability
        column_names = [description[0] for description in cursor.description]
        
        # Format results as list of dictionaries
        formatted_results = []
        for row in results:
            formatted_row = {}
            for i, value in enumerate(row):
                formatted_row[column_names[i]] = value
            formatted_results.append(formatted_row)
        
        return {
            "valid": True,
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
