
def get_prompt(tables_info, table_samples, library):
    """
    1. Generates a question_prompt string for an LLM to produce an SQL-relevant
       natural language question.
    2. Returns a second string, sql_prompt, for the next step (SQL translation),
       with all necessary context pre-populated.
    """

    # --- Start building the prompt for question generation ---
    question_prompt = f"""
You are an SQL **question generator** in natural language for a database with the following structure.

DATABASE TABLES:
{', '.join(tables_info['tables'])}

SCHEMA FOR EACH TABLE:
"""

    sql_prompt = f"""
You are an SQL **expert** translating natural language question to SQL.

DATABASE TABLES:
{', '.join(tables_info['tables'])}

SCHEMA FOR EACH TABLE:
"""

    # -- Add table + column info to both prompts --
    for table, columns in tables_info['schemas'].items():
        question_prompt += f"\nTable: {table}\n"
        sql_prompt += f"\nTable: {table}\n"
        for column in columns:
            question_prompt += f"  - {column}\n"
            sql_prompt += f"  - {column}\n"

    # -- Add sample data to both prompts --
    question_prompt += "\nSAMPLE DATA:\n"
    sql_prompt += "\nSAMPLE DATA:\n"
    for table, rows in table_samples.items():
        question_prompt += f"\nTable: {table} (showing {len(rows)} rows)\n"
        sql_prompt += f"\nTable: {table} (first few rows):\n"
        for i, row in enumerate(rows):
            question_prompt += f"  Row {i+1}: {row}\n"
            sql_prompt += f"  {row}\n"

    # -- Optional: Add library context to the question_prompt --
    if library:
        question_prompt += f"\nNOTE: The library already contains {len(library)} questions."
        question_prompt += "\nRecent questions in the library (up to 3 most recent):"
        for i in range(min(3, len(library))):
            idx = len(library) - i - 1
            question_prompt += f"\n- {library[idx]['question']}"
        
        if len(library) > 0:
            question_prompt += "\n\nYour NEW question should be similar in complexity but *not* a duplicate. " \
                                "Try to query different tables or relationships to avoid exact overlap."

    # -- Final instructions for the LLM to produce the question in natural language --
    question_prompt += """
                    Using the database schema and sample data above(You Do NOT have an access the sample data provided in the task description), generate a clear and specific
                    **natural language question**.

                    IMPORTANT CRITERIA:
                    1. The question should be specific enough to be translated into SQL.
                    2. The question must have an answer in the database based on the provided sample above 
                    3. Don't code the table.
                    3. Write it in plain, clear natural language without referencing code.
                    4. You must use the "retriever_tool" tool it validates the generated question for you. 
                    5. Finally retrun the natural language question. "final_answer(retriever_tool(question))"

                    Return **only the question** (no explanations or extra formatting).
                    """

    return question_prompt, sql_prompt
def get_extra_prompt_sql(sql_prompt, question):
    """
    2. Takes the partial 'sql_prompt' plus the new 'question' generated
       in natural language, and instructs the LLM to produce the SQL query.
    """
    
    sql_prompt += f"""
                Natural Language Question: {question}

                INSTRUCTION:
                Write a one SINGLE valid SQL query that correctly answers Natural Language Question using the
                database structure provided above. JOIN tables if needed.
                And you must use the "execute_sql" tool to check if the sql is valid.
                Finally Return the valid SQL. "final_answer(execute_sql(sql_query))"

                RETURN ONLY the SQL query without explanations, commentary, or markdown formatting.
                """
    return sql_prompt

def get_extra_prompt_divers(question, sql_question, tables_info):
    """
    3. Takes the original question, the final SQL query, and table schema info
       to generate 3 paraphrased variations that preserve meaning and use the
       same columns/tables.
    """

    diversity_prompt = f"""
    You are a **question paraphrasing expert**.

    Original question: {question}
    its Corresponding SQL query: {sql_question}

    DATABASE TABLES Columns:
    {tables_info["schemas"].values()}


        Your task is to create 7 different variants of the original question using the following techniques:

    1. BASIC SIMPLIFICATION
    • Remove non-essential elements while preserving the main meaning.
    • Example: "Which club has the most female students as their members?" → "Which club has the most female students?"

    2. SIMPLIFICATION BY HIDING DETAILS
    • Preserve the central objective but remove additional explanations.
    • Example: "What is the title and credits of the course that is taught in the largest classroom?" → "What course is taught in the biggest classroom and what are its credits?"

    3. USING SYNONYMS
    • Replace certain terms with their synonyms while maintaining meaning.
    • Example: "What is the average duration in milliseconds of tracks that belong to Latin or Pop genre?" → "What is the mean length in milliseconds of Latin or Pop songs?"

    4. SEMANTIC SUBSTITUTIONS
    • Replace expressions with semantically equivalent alternatives.
    • Example: "What are the locations that have gas stations owned by a company with a market value greater than 100?" → "Where are the gas stations owned by a company worth more than 100?"

    5. COMPLETE REFORMULATION
    • Express the same request in a totally different way.
    • Example: "What is the number of routes operated by American Airlines whose destinations are in Italy?" → "How many routes does American Airlines have that fly to Italy?"

    6. SIMPLIFICATION WITH MORE DIRECT LANGUAGE
    • Use more direct and concise language.
    • Example: "What are the names of body builders whose total score is higher than 300?" → "Who are the body builders with a score over 300?"

    7. PARAPHRASE WITH CHANGE OF PERSPECTIVE
    • Reformulate by changing the style or perspective of the question.
    • Example: "Return the categories of music festivals that have the result 'Awarded'" → "List the categories of music festivals that have been recognized with awards"

    For each original SQL question provided, please generate all 7 variants following these techniques without changing the sql request.
    Make sure to keep the SQL query unchanged.

    tools:
    - Use 'get_synonym' tool to look online for synonyms if need.
    - You must use the "retriever_tool" tool for each variation question to validate the generated question for you.

    return a json array with the following format (no additional text):
    [
        {{
            "question": "Variant 1",
            "sql": "SQL Query"
        }},
        {{
            "question": "Variant 2",
            "sql": "SQL Query"
        }},
        ...
        {{
            "question": "Variant 7",
            "sql": "SQL Query"
        }}
    ]
    """
    return diversity_prompt


"""
INSTRUCTION:
- Create 3 alternative phrasings of the *original question* that would be answered
  by the same SQL query.
- Do NOT change any table or column names.
- Use synonyms and rephrasings to vary the language, but preserve the question's meaning.
- Return each variation as a separate line or list entry.

Techniques to apply:
1. Simplify by hiding or restructuring details (but keep the question accurate).
2. Use synonyms or short paraphrases where possible.
3. Express it differently in terms of word order or phrasing.
4. Maintain the essential meaning so the same SQL query still applies.
5. Words in quotes ' ' must not be modified.
6. Use 'get_synonym' tool to look online for synonyms if need.
7. You must use the "retriever_tool" tool for each variation question to validate the generated question for you.
8. final_answer a list of 3 new generated questions. 



Return exactly 3 rephrased questions in a list-like format, no additional text.
"""


