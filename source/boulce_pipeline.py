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
from tools import get_synonym,ExecuteSQLTool,get_table
from prompt import get_extra_prompt_divers,get_prompt,get_extra_prompt_sql

from langchain_community.vectorstores import FAISS



# Initialize model based on available API keys
if not os.environ.get("OPENAI_API_KEY"):
    #model = HfApiModel()
    model = LiteLLMModel(
        #deepseek-r1:14b /llama3.1:8b-instruct-fp16 ollama run qwq:32b-fp16
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
def create_agents(model,retriever_tool,execute_sql):
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
        additional_authorized_imports=["numpy"],
    )

    sql_translator = CodeAgent(
        model=model,
        tools=[execute_sql],
        name="sql_translator", 
        description="Translates natural language questions into SQL queries",
        additional_authorized_imports=["pandas","numpy","time"],#,"sqlite3"
    )

    question_diversity = CodeAgent(
        model=model,
        tools=[get_synonym],
        name="question_diversity",
        description="Creates diverse variations of questions using different techniques",
        additional_authorized_imports=["pandas","numpy","time"],
    )
    
    return question_generator, sql_translator, question_diversity


def run_pipeline_step(question_prompt:str,sql_prompt:str,tables_info:str, 
                            question_generator, sql_translator, question_diversity,
                            retriever_tool,execute_sql, max_attempts: int = 5,)-> Optional[Dict[str, Any]]:
    """Run the full pipeline once to generate a dataset entry with validation"""
    for attempt in range(max_attempts):
        print(f"Attempt {attempt+1}/{max_attempts} to generate a valid entry...")
      
    
        question = question_generator.run(question_prompt)
        print(f"Generated question: {question}")

        sql_prompt=get_extra_prompt_sql(sql_prompt,question)
        
        
        sql_query = sql_translator.run(sql_prompt)
        print(f"Generated SQL: {sql_query}")
        validation_result=execute_sql(sql_query)
        
        # 3. Validate the SQL query

            
        print(f"SQL validation successful! Found {validation_result} results.")
        

        # 5. Generate question variations
        entry = []
        if "Error executing SQL" in str(validation_result) or len(validation_result)==0: continue
        diversity_prompt=get_extra_prompt_divers(question,sql_query,tables_info)
        variations = question_diversity.run(diversity_prompt)
        print(f"Generated variations: {variations}")
        
        retriever_tool.vectordb.add_documents(documents=[Document(question)], ids=[str(uuid4())])
        entry.append({
                "tables": tables_info['tables'],
                "question": question,
                "sql": sql_query,
                "result": validation_result,
            })
        for varaition in variations:
            retriever_tool.vectordb.add_documents(documents=[Document(varaition)], ids=[str(uuid4())])
            entry.append({
                "tables": tables_info['tables'],
                "question": varaition,
                "sql": sql_query,
                "result": validation_result #validation_result["results"],
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


    conn,tables_info,table_samples = get_table(db_path)
    question_prompt,sql_prompt=get_prompt(tables_info,table_samples,library)
    execute_sql= ExecuteSQLTool(conn)  

    # Create agents
    question_generator, sql_translator, question_diversity = create_agents(model,retriever_tool,execute_sql)
    
    # Main generation loop
    progress_bar = tqdm(range(num_entries), desc="Generating dataset entries")
    for i in progress_bar:
        progress_bar.set_description(f"Entry {len(library) + 1}")
        
        # Run one pipeline step
        entries = run_pipeline_step(question_prompt,sql_prompt,tables_info, 
                                    question_generator, sql_translator, question_diversity,
                                    retriever_tool,execute_sql)
        
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
    #db_path = "../data/tables/db/{}.db"  # Path to the database file
    db_path ="../data/tables/db/200_0.db"
    num_entries = 10  # Number of entries to generate
    
    generate_dataset(db_path, num_entries)


