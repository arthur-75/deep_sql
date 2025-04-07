
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
                    5. Finally retrun the natural language question. final_answer(retriever_tool(question))

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
                Finally Return the valid SQL.

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
Corresponding SQL query: {sql_question}

DATABASE TABLES Columns:
{tables_info["schemas"].values()}

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


Return exactly 3 rephrased questions in a list-like format, no additional text.
"""
    return diversity_prompt
""