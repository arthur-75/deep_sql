import os
import random
import pandas as pd
import sqlite3

class TableManager:
    """
    TableManager is responsible for retrieving and describing tables from a given database directory.
    Supports CSV, SQLite, and Parquet formats.
    """

    def __init__(self, db_directory: str):
        """
        Initialize the TableManager with the database directory.
        
        :param db_directory: Path to the directory containing database tables.
        """
        self.db_directory = db_directory
        self.table_paths = self._get_table_paths()

    def _get_table_paths(self) -> list:
        """
        Retrieve all file paths from the database directory.

        :return: List of full file paths of tables.
        """
        if not os.path.exists(self.db_directory):
            raise FileNotFoundError(f"Database directory '{self.db_directory}' does not exist.")

        return [
            os.path.join(self.db_directory, f) 
            for f in os.listdir(self.db_directory) 
            if os.path.isfile(os.path.join(self.db_directory, f))
        ]

    def get_random_table(self) -> str:
        """
        Select a random table file from the database directory.

        :return: The file path of the selected table.
        """
        if not self.table_paths:
            raise ValueError("No tables found in the database directory.")
        
        return random.choice(self.table_paths)

    def get_table_info(self, database_file: str) -> str:
        """
        Return a formatted description of a table for prompting.

        :param database_file: Path to the SQLite database file.
        :return: String containing the table name and schema details.
        """
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()

        # Get table names
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        if not tables:
            conn.close()
            raise ValueError(f"No tables found in the database: {database_file}")

        table_name = tables[0][0]  # Load the first table (modify if needed)

        # Get column metadata
        columns_info = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
        conn.close()

        # Format description
        table_description = f"📋 **Table Name:** {table_name}\n"
        table_description += "(column_index, column_name, data_type, not_null, default_value, primary_key)\n"
        for col in columns_info:
            table_description += f"{col}\n"

        return table_description

    def get_random_table_info(self) -> str:
        """
        Select a random table and return its description.

        :return: Formatted table schema description.
        """
        try:
            table_path = self.get_random_table()
            return self.get_table_info(table_path), table_path
        except Exception as e:
            return f"❌ Error retrieving table info: {e}"