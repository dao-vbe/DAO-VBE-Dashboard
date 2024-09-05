"""
This module contains the DatabaseHandler class which provides methods for interacting with the PostgreSQL database.

Example usage:
    config = {
        'user': 'my_user',
        'password': 'my_password',
        'host': 'localhost',
        'database': 'my_database'
    }
    handler = DatabaseHandler(config)
    handler.write_to_sql(df, 'my_table')

"""
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2 import sql

load_dotenv()

class DatabaseHandler:
    DEFAULT_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    def __init__(self, custom_config=None):
        """
        Initializes a DatabaseHandler object.

        Args:
            config (dict): A dictionary containing the configuration parameters for the database connection.
                The dictionary should include the following keys: 'user', 'password', 'host', and 'database'.
        """
        self.config = custom_config if custom_config else self.DEFAULT_CONFIG
        self.engine = create_engine(f"postgresql+psycopg2://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}")
        print("DatabaseHandler init successful")

    def write_to_sql(self, df, table_name, if_exists='append', index=True):
        """
        Writes a pandas DataFrame to a SQL table.

        Args:
            df (pandas.DataFrame): The DataFrame to be written to the SQL table.
            table_name (str): The name of the SQL table.
            if_exists (str, optional): Specifies how to behave if the table already exists.
                Possible values are 'fail', 'replace', and 'append'. Defaults to 'append'.
            index (bool, optional): Specifies whether to include the DataFrame index as a column in the SQL table. Defaults to True.
        """
        print(df.tail())
        df.to_sql(table_name, self.engine, index=index, if_exists=if_exists)
        print(f"Data written to table {table_name}")


    def db_to_df(self, sql_table):
        """
        Retrieve data from the specified SQL table and return it as a pandas DataFrame.

        Parameters:
        sql_table (str): The name of the SQL table to retrieve data from.

        Returns:
        pandas.DataFrame: A DataFrame containing the retrieved data.

        """
        connection = psycopg2.connect(
            dbname=self.config['database'],
            user=self.config['user'],
            password=self.config['password'],
            host=self.config['host'],
            port=self.config['port']
        )
        cursor = connection.cursor()
        cursor.execute("SET search_path TO public;")
        query = f"""SELECT * FROM {sql_table};"""
        cursor.execute(query)
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(records, columns=columns)
        cursor.close()
        connection.close()
        return df


    def df_to_sql(self, df, table_name, if_exists='append'):
        df.to_sql(table_name, self.engine, schema='public', if_exists=if_exists, index=False)

    def get_schema(self, table_name):
        query = f"""SELECT 
                    column_name, 
                    data_type, 
                    is_nullable, 
                    column_default
                FROM 
                    information_schema.columns
                WHERE 
                    table_schema = 'public' 
                    AND table_name = '{table_name}';
                 """
        return self.execute_query(query)

    # Execute SQL query on the database
    def execute_query(self, query, print_output=True):
        """
        Executes a SQL query on the database and returns the result as a pandas DataFrame.

        Args:
            query (str): The SQL query to execute.
            print_output (bool, optional): Whether to print the output to the console. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the query results.
        """
        connection = psycopg2.connect(
            dbname=self.config['database'],
            user=self.config['user'],
            password=self.config['password'],
            host=self.config['host'],
            port=self.config['port']
        )
        cursor = connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(records, columns=columns)
        if print_output:
            print(df)
        cursor.close()
        connection.close()
        return df
