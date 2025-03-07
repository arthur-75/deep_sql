import re

def extract_sql_from_text(text: str) -> str:
    """
    Extracts SQL code enclosed within triple backticks from a given text.

    :param text: The input string containing SQL code.
    :return: Extracted SQL query as a string, or an empty string if no SQL is found.
    """
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)  # DOTALL makes '.' match newlines
    
    return match.group(1).strip() if match else ""