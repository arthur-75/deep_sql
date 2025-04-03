import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import json
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from smolagents import CodeAgent, tool, HfApiModel, OpenAIServerModel,LiteLLMModel
#from evaluate import load
#bertscore = load("bertscore")
from retriever import embeddings_vector_store,RetrieverTool
from uuid import uuid4
from langchain_core.documents import Document
import pickle



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
    # Convert to string if necessary
    if isinstance(sql1, list):
        sql1 = str(sql1)
    if isinstance(sql2, list):
        sql2 = str(sql2)
        
    #vectorizer = TfidfVectorizer()
    #tfidf_matrix = vectorizer.fit_transform([sql1, sql2])
    #return float((tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0])
    score = bertscore.compute(predictions=[sql1], references=[sql2], lang="en")
    #score = bertscore.compute([sql1], [sql2])[2].item()
    print("BS: ", score)
    return score['f1'][0]

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
    
    max_similarity = min(all_similarities)
    recent_similarity = all_similarities[-1]
    
    if max_similarity > gamma_max:
        return (False, f"Too similar to existing query (similarity: {max_similarity:.2f})")
    
    if recent_similarity < gamma_min:
        return (False, f"Too different from recent query (similarity: {recent_similarity:.2f})")
    
    return (True, f"Query accepted (max similarity: {max_similarity:.2f}, recent similarity: {recent_similarity:.2f})")





# Initialize model based on available API keys
if not os.environ.get("OPENAI_API_KEY"):
    #model = HfApiModel()
    model = LiteLLMModel(
    model_id="ollama_chat/llama3.1:8b-instruct-fp16", # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    num_ctx=8192,device="mps" # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
    )   
else:
    model = OpenAIServerModel("gpt-4o")

# Initialize the dataset library
def init_library(library_path: str = "sql_dataset_library.json",vector_store_path='vector_store.pkl') -> List[Dict[str, Any]]:
    """Initialize or load the existing library"""
    if os.path.exists(library_path):
        with open(library_path, 'r') as f:
            library= json.load(f)
        with open(vector_store_path, 'rb') as handle:
            vector_store = pickle.load(handle)
        return library,vector_store
   
    return [],embeddings_vector_store()

# Save the library to disk
def save_library(library: List[Dict[str, Any]],vector_store, vector_store_path='vector_store.pkl',library_path: str = "sql_dataset_library.json",):
    """Save the current library to disk"""
    with open(library_path, 'w') as f:
        json.dump(library, f, indent=2)
    with open(vector_store_path, 'wb') as handle:
        pickle.dump(vector_store, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create the agents with access to appropriate tools
def create_agents(model, library,retriever_tool):
    """Create the pipeline agents with access to the library"""
    
    question_generator = CodeAgent(
        model=model,
        tools=[
            #get_tables_info,
            #get_table_samples,
            #check_query_novelty
            retriever_tool
        ],
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
    
    return question_generator, sql_translator, question_diversity

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

def run_pipeline_step(db_path: str, library: List[Dict[str, Any]], 
                      question_generator, sql_translator, question_diversity,retriever_tool,
                      max_attempts: int = 5,) -> Optional[Dict[str, Any]]:
    """Run the full pipeline once to generate a dataset entry with validation"""
    
    conn = connect_to_database(db_path)
    tables_info = get_tables_info(conn)
    table_samples = get_table_samples(conn)
    
    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1}/{max_attempts} to generate a valid entry...")
        
        # 1. Generate question with context
        question_prompt = f"""
        You are an SQL question generator for a database with the following structure:

        DATABASE TABLES:
        {', '.join(tables_info['tables'])}

        SCHEMA FOR EACH TABLE:
        """

        for table, columns in tables_info['schemas'].items():
            question_prompt += f"\nTable: {table}\n"
            for column in columns:
                question_prompt += f"  - {column}\n"

        question_prompt += "\nSAMPLE DATA:\n"
        for table, rows in table_samples.items():
            question_prompt += f"\nTable: {table} (showing {len(rows)} rows)\n"
            for i, row in enumerate(rows):
                question_prompt += f"  Row {i+1}: {row}\n"
        
        # Add library context if available
        if library:
            question_prompt += f"\nNOTE: The library already contains {len(library)} questions."
            question_prompt += "\nRecent questions in the library:"
            for i in range(min(3, len(library))):
                idx = len(library) - i - 1
                question_prompt += f"\n- {library[idx]['question']}"
            
            if len(library) > 0:
                question_prompt += "\n\nYour question should be similar to the existing ones in complexity, but ask about different tables or relationships."
        
        question_prompt += """
        Using the database schema and sample data above, generate a clear, specific question 
        that can be answered using SQL on this database.
        
        IMPORTANT: Make sure your question:
        1. Is specific enough to be translated into SQL
        2. Has an answer in the database (based on the sample data)
        3. Requires only one SQL query to answer
        4. Is written in simple, clear language
        
        Return only the question without any explanations.
        """
        
        question = question_generator.run(question_prompt)
        print(f"Generated question: {question}")
        
        # 2. Translate to SQL
        sql_prompt = f"""
        You are an SQL expert translating natural language questions to SQL.
        
        DATABASE TABLES:
        {', '.join(tables_info['tables'])}

        SCHEMA FOR EACH TABLE:
        """

        for table, columns in tables_info['schemas'].items():
            sql_prompt += f"\nTable: {table}\n"
            for column in columns:
                sql_prompt += f"  - {column}\n"

        sql_prompt += "\nSAMPLE DATA:\n"
        for table, rows in table_samples.items():
            sql_prompt += f"\nTable: {table} (first few rows):\n"
            for i, row in enumerate(rows[:3]):  # Only show first 3 rows for brevity
                sql_prompt += f"  {row}\n"
        
        sql_prompt += f"""
        Question: {question}
        
        Write a SINGLE valid SQL query that correctly answers this question.
        Consider JOINs between tables if needed.
        Make sure the query will return results based on the sample data shown.
        Return ONLY the SQL query without any explanations or markdown formatting.
        """
        
        sql_query = sql_translator.run(sql_prompt)
        print(f"Generated SQL: {sql_query}")
        
        # 3. Validate the SQL query
        validation_result = validate_sql_query(conn, sql_query)
        
        if not validation_result["valid"]:
            print(f"SQL validation failed: {validation_result.get('error', 'Unknown error')}")
            continue  # Try again with a new question
            
        print(f"SQL validation successful! Found {validation_result['row_count']} results.")
        
        # 4. Check novelty if library exists
        #if library:
        #    is_novel, message = check_query_novelty(library, sql_query)
        #    print(f"Novelty check: {message}")
        #    if not is_novel:
        #        print("Query did not meet novelty requirements, retrying...")
        #        continue  # Try again with a new question
        
        # 5. Generate question variations
        diversity_prompt = f"""
        You are a question paraphrasing expert.
        Original question: {question}
        
        Create 3 variations of this question that would be answered by the same SQL query.
        
        Use these techniques from the reference table:
        1. Simplify by hiding details
        2. Simplify using synonyms
        3. Express in a different way
        
        Return your response as a list of 3 questions, clearly numbered 1, 2, and 3.
        Make sure each variation preserves the original meaning and would be answered by the same SQL query.
        """
        
        variations = question_diversity.run(diversity_prompt)
        print(f"Generated variations: {variations}")
        entry = []
        retriever_tool.vectordb.add_documents(documents=[Document(question)], ids=[str(uuid4())])
        entry.append({
                "tables": tables_info['tables'],
                "question": question,
                "sql": sql_query,
                "result": validation_result["results"],
            })
        for varaition in variations:
            retriever_tool.vectordb.add_documents(documents=[Document(varaition)], ids=[str(uuid4())])
            entry.append({
                "tables": tables_info['tables'],
                "question": varaition,
                "sql": sql_query,
                "result": validation_result["results"],
            })
        
        return entry
    
    # If we've exhausted all attempts
    print("Failed to generate a valid entry after multiple attempts.")
    return None


# The main loop for dataset generation
def generate_dataset(db_path: str, num_entries: int, library_path: str = "sql_dataset_library.json") -> None:
    """
    Generate a dataset with the specified number of entries
    
    Args:
        db_path: Path to the SQLite database
        num_entries: Number of entries to generate
        library_path: Path to save the library JSON
    """
    # Initialize or load existing library
    library,vector_store = init_library(library_path)
    print(f"Starting with library containing {len(library)} entries")
    
    retriever_tool = RetrieverTool(vector_store)

    # Create agents
    question_generator, sql_translator, question_diversity = create_agents(model, library,retriever_tool)
    
    # Main generation loop
    progress_bar = tqdm(range(num_entries), desc="Generating dataset entries")
    for i in progress_bar:
        progress_bar.set_description(f"Entry {len(library) + 1}")
        
        # Run one pipeline step
        entries = run_pipeline_step(db_path, library, question_generator, sql_translator, question_diversity,retriever_tool)
        
        if entries:
            # Add to library
            for entry in entries:
                library.append(entry)
            print(f"Added entry #{len(library)} to library")
            
            # Save after each successful addition
            save_library(library, library_path)
            
            progress_bar.set_postfix(library_size=len(library))
        else:
            print("Failed to generate entry, continuing...")
    
    print(f"Dataset generation complete. Final library size: {len(library)}")

# Example usage
if __name__ == "__main__":
    db_path = "../data/tables/db/200_0.db"  # Path to the database file
    num_entries = 3  # Number of entries to generate
    
    generate_dataset(db_path, num_entries)