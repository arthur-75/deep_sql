import sqlite3
from typing import Dict, List, Tuple, Any, Union, Optional
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from retriever import embeddings_vector_store,get_emebdding_model
import json
import os


def get_table_dirty(path:str=None):
    # Load squall
    if path is None:
        file_path = "/Users/arthur/Documents/reasearch/deep_sql/data/squall.json"

    with open(file_path, 'r') as json_file:
        squall = json.load(json_file)

    # Extract unique squall IDs
    squall_ids = set(i["nt"] for i in squall)

    # Load WTQ
    wtq = load_dataset('wikitablequestions')
    wtq_train = wtq["train"]

    # Extract WTQ IDs
    wtq_ids = set(wtq_train["id"])

    # Get intersection of IDs
    common_ids = list(squall_ids.intersection(wtq_ids))

    # Mapping from ID to table for squall
    squall_table_id_by_id = {entry["nt"]: entry["tbl"] for entry in squall if entry["nt"] in common_ids}

    # Mapping from ID to table for wtq
    wtq_table_by_id = {entry["id"]: entry["table"] for entry in wtq_train if entry["id"] in common_ids}

    return squall_table_id_by_id, wtq_table_by_id, common_ids
    


def get_table(db_path):
    conn = connect_to_database(db_path)
    tables_info = get_tables_info(conn)
    table_samples = get_random_table_samples(conn)

    return conn,tables_info,table_samples


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



# Initialize the dataset library
def init_library(library_path: str = "sql_dataset_library.json",vector_store_path='vector_store',model_name="Alibaba-NLP/gte-large-en-v1.5") -> List[Dict[str, Any]]:
    """Initialize or load the existing library"""
    if os.path.exists(library_path):
        with open(library_path, 'r') as f:
            library= json.load(f)
        vector_store = FAISS.load_local(vector_store_path, embeddings=get_emebdding_model(model_name) , allow_dangerous_deserialization=True)
        return library,vector_store
   
    return [],embeddings_vector_store(model_name)

# Save the library to disk
def save_library(library: List[Dict[str, Any]],vector_store,library_path: str = "sql_dataset_library.json", vector_store_path='vector_store',):
    """Save the current library to disk"""
    with open(library_path, 'w') as f:
        json.dump(library, f, indent=2)
    vector_store.save_local(vector_store_path)