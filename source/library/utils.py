

import sqlite3


def get_table_info(database_name:str='database.sqlite')-> str:
    """
    return description of dataset
    """
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    table_name=cursor.execute("SELECT name FROM sqlite_master").fetchall()[0][0]
    columns_info = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
    table_description=f"Table name :{table_name}\n(column_index, column_name, data_type, not_null, default_value, primary_key)\n{columns_info}"
    return table_description