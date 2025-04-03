import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from smolagents import CodeAgent, tool, HfApiModel

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
def calculate_similarity(sql1: str, sql2: str) -> float:
    """
    Calculate cosine similarity between two SQL queries.
    
    Args:
        sql1: First SQL query string
        sql2: Second SQL query string
    
    Returns:
        Similarity score between 0 and 1
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sql1, sql2])
    return float((tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0])

@tool
def check_query_novelty(library: List[Dict[str, Any]], new_sql: str, gamma_max: float = 0.9, gamma_min: float = 0.6) -> Tuple[bool, str]:
    """
    Check if the new SQL query meets the similarity constraints.
    
    Args:
        library: List of existing dataset entries
        new_sql: New SQL query to check
        gamma_max: Maximum allowed similarity threshold
        gamma_min: Minimum required similarity threshold
    
    Returns:
        Tuple of (is_novel, explanation_message)
    """
    if not library:
        return (True, "First query - automatically accepted")
    
    all_similarities = []
    for entry in library:
        similarity = calculate_similarity(entry["sql"], new_sql)
        all_similarities.append(similarity)
    
    max_similarity = max(all_similarities)
    recent_similarity = all_similarities[-1]
    
    if max_similarity > gamma_max:
        return (False, f"Too similar to existing query (similarity: {max_similarity:.2f})")
    
    if recent_similarity < gamma_min:
        return (False, f"Too different from recent query (similarity: {recent_similarity:.2f})")
    
    return (True, f"Query accepted (max similarity: {max_similarity:.2f}, recent similarity: {recent_similarity:.2f})")




# Initialize model and library
#model = HfApiModel()
from smolagents import OpenAIServerModel

# Replace the previous model initialization with this:
# check if OPENAI_API_KEY is set
import os

if not os.environ.get("OPENAI_API_KEY"):
    model = HfApiModel()
else:
    model = OpenAIServerModel("gpt-4o")


library = []

question_generator = CodeAgent(
    model=model,
    tools=[get_tables_info, get_table_samples],
    name="question_generator",
    description="Generates a new database question based on schema and sample data"
)


sql_translator = CodeAgent(
    model=model,
    tools=[execute_sql],
    name="sql_translator", 
    description="Translates natural language questions into SQL queries"
)

question_diversity = CodeAgent(
    model=model,
    tools=[],
    name="question_diversity",
    description="Creates diverse variations of questions using different techniques"
)

# Main pipeline function
def run_pipeline(db_path: str) -> Optional[Dict[str, Any]]:
    """Run the full pipeline once to generate a dataset entry"""
    print("1. Connecting to database...")
    conn = connect_to_database(db_path)
    
    
    # Then in run_pipeline function, modify this part:
    print("2. Getting table information and samples...")
    tables_info = get_tables_info(conn)
    table_samples = get_table_samples(conn)


    # system_prompt_step = question_generator.memory.system_prompt
    # print("The system prompt given to the agent was:")
    # print(system_prompt_step.system_prompt)

    # exit()

    print("3. Generating question...")
    question_prompt = f"""
    You are an SQL question generator.
    Database tables: {tables_info['tables']}
    Table schemas: {tables_info['schemas']}

    Sample data from tables:
    {table_samples}

    Using the database schema and sample data above, generate a clear, focused question that can be answered using SQL on this database.
    Make sure your question is specific and can be accurately answered based on the sample data shown.
    Return only the question without any explanations.
    IMPORTANT: You must respond with Python code that returns the question as a string. regex pattern ```(?:py|python)?\n(.*?)\n``` is needed to extract the response.

    """
    
    question_prompt = f"""
    You are an SQL question generator for a database with the following structure:

    DATABASE TABLES:
    {', '.join(tables_info['tables'])}

    SCHEMA FOR EACH TABLE:
    """

    # Add schema information in a structured way
    for table, columns in tables_info['schemas'].items():
        question_prompt += f"\nTable: {table}\n"
        for i, column in enumerate(columns):
            question_prompt += f"  - {column}\n"

    question_prompt += "\nSAMPLE DATA:\n"
    limit = 100
    # Add sample data in a structured way
    for table, rows in table_samples.items():
        question_prompt += f"\nTable: {table} (showing {len(rows)} of {limit} rows)\n"
        for i, row in enumerate(rows):
            question_prompt += f"  Row {i+1}: {row}\n"

    question_prompt += """
    Using the database schema and sample data above, generate a clear, focused question that can be answered using SQL on this database.
    Start with simple question. Make sure your question is specific and can be accurately answered based on the sample data shown.
        """

    
    question = question_generator.run(question_prompt)
    print(f"Generated question: {question}")
    
    
    print("4. Translating to SQL...")
    sql_prompt = f"""
    You are an SQL expert.
    Database tables: {tables_info['tables']}
    Table schemas: {tables_info['schemas']}
    
    Translate this question into a valid SQL query:
    Question: {question}
    
    Return only the SQL query without any explanations.
    IMPORTANT: You must respond with Python code that returns the SQL query as a string. regex pattern ```(?:py|python)?\n(.*?)\n``` is needed to extract the response.

    """
    sql_query = sql_translator.run(sql_prompt)
    print(f"Generated SQL: {sql_query}")
    
    print("5. Executing SQL to get result...")
    result = execute_sql(conn, sql_query)
    print(f"Query result: {result}")
    
    print("6. Checking query novelty...")
    if library:  # Skip this step for the first query
        is_novel, message = check_query_novelty(library, sql_query)
        print(f"Novelty check: {message}")
        if not is_novel:
            print("Query did not meet novelty requirements, stopping here.")
            return None
    
    print("7. Generating question variations...")
    diversity_prompt = f"""
    You are a question paraphrasing expert.
    Original question: {question}
    
    Create 3 variations of this question using these techniques from the reference table:
    1. Simplify by hiding details
    2. Simplify using synonyms
    3. Express in a different way
    
    Format your response as a list of the 3 variations.
    """
    variations = question_diversity.run(diversity_prompt)
    print(f"Generated variations: {variations}")
    
    # Add to library
    entry = {
        "tables": tables_info['tables'],
        "question": question,
        "sql": sql_query,
        "result": result,
        "variations": variations
    }
    library.append(entry)
    
    print("8. Entry added to library")
    return entry

# Example usage
if __name__ == "__main__":
    db_path = "../data/tables/db/200_0.db"  # Path to the database file
    entry = run_pipeline(db_path)
    
    if entry:
        print("\nGenerated Dataset Entry:")
        print(f"Question: {entry['question']}")
        print(f"SQL Query: {entry['sql']}")
        print(f"Result: {entry['result']}")
        print(f"Variations: {entry['variations']}")