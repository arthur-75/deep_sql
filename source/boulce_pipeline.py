import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import json
import os
from tqdm import tqdm
from smolagents import CodeAgent, HfApiModel, OpenAIServerModel,LiteLLMModel
#from evaluate import load
#bertscore = load("bertscore")
from retriever import embeddings_vector_store,RetrieverTool,get_emebdding_model
from uuid import uuid4
from langchain_core.documents import Document
from tools import get_synonym,execute_sql,get_table_samples,get_tables_info,connect_to_database,validate_sql_query
from langchain_community.vectorstores import FAISS



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
        description="Generates a new database question based on schema and sample data",
        additional_authorized_imports=["pandas","numpy"],
    )

    sql_translator = CodeAgent(
        model=model,
        tools=[execute_sql],
        name="sql_translator", 
        description="Translates natural language questions into SQL queries",
        additional_authorized_imports=["pandas","numpy"],#,"sqlite3"
    )

    question_diversity = CodeAgent(
        model=model,
        tools=[get_synonym],
        name="question_diversity",
        description="Creates diverse variations of questions using different techniques",
        additional_authorized_imports=["pandas","numpy","sqlite3","requests","bs4"],
    )
    
    return question_generator, sql_translator, question_diversity


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
def generate_dataset(db_path: str, num_entries: int, library_path: str = "sql_dataset_library.json",vector_store_path='vector_store') -> None:
    """
    Generate a dataset with the specified number of entries
    
    Args:
        db_path: Path to the SQLite database
        num_entries: Number of entries to generate
        library_path: Path to save the library JSON
    """
    # Initialize or load existing library
    library,vector_store = init_library(library_path,vector_store_path)
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
            save_library(library,vector_store, library_path,vector_store_path)
            
            progress_bar.set_postfix(library_size=len(library))
        else:
            print("Failed to generate entry, continuing...")
    
    print(f"Dataset generation complete. Final library size: {len(library)}")

# Example usage
if __name__ == "__main__":
    db_path = "../data/tables/db/200_0.db"  # Path to the database file
    num_entries = 3  # Number of entries to generate
    
    generate_dataset(db_path, num_entries)