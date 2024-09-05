"""
This module provides utility functions to interact with a PostgreSQL database.

The functions in this module allow for the creation of tables,
insertion of data, and retrieval of data from the database.

Note: Ensure that the required environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) are set before using this module.
"""

import csv
import os
from dotenv import load_dotenv
import psycopg2
import requests
import database as db

# Load environment variables from .env file
load_dotenv()

# Database connection
db_params = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}
csv_file_path = 'data_input/dao_input.csv'

# Create table and import data function
def clean_numeric(value):
    """Clean and convert a string to a numeric value, return None if not possible."""
    if value is None or value.strip() == '':
        return None
    try:
        return float(value.replace(',', '').strip())
    except ValueError:
        print(f"Warning: Could not convert '{value}' to a number. Setting to None.")
        return None

# Function to create table and import data
def create_tables(cur):
    create_table_sql = """
    CREATE TABLE proposals (
        id SERIAL PRIMARY KEY,
        platform VARCHAR(255),
        dao_id VARCHAR(255),
        protocol VARCHAR(255)
        proposal_id VARCHAR(255),
        proposal_title TEXT,
        proposal_body TEXT,
        choices TEXT,
        start_date VARCHAR(255),
        end_date VARCHAR(255),
        created VARCHAR(255),
        state VARCHAR(255),
        choice_scores VARCHAR(255),
        quorum NUMERIC,
        creator VARCHAR(255),
        proposer VARCHAR(255),
        discussion TEXT
    );

    CREATE TABLE dao (
        id SERIAL PRIMARY KEY,
        platform VARCHAR(255),
        dao_id VARCHAR(255),
        dao_name VARCHAR(255),
        dao_slug VARCHAR(255),
        about VARCHAR(255),
        protocol VARCHAR(255),
        category VARCHAR(255),
        treasury INTEGER,
        member_count INTEGER,
        proposal_count INTEGER,
        active_voters INTEGER,
        votes_cast VARCHAR(255)
    );

    CREATE TABLE votes (
        id SERIAL PRIMARY KEY,
        platform VARCHAR(255),
        proposal_id VARCHAR(255),
        vote_id VARCHAR(255),
        voter_address VARCHAR(255),
        voter_name VARCHAR(255),
        choice VARCHAR(255),
        voting_power NUMERIC,
        reason TEXT,
        discussion VARCHAR(2000)
    );

    CREATE TABLE proposal_stats (
        id SERIAL PRIMARY KEY,
        proposal_id VARCHAR(255),
        choice VARCHAR(255),
        measure TEXT,
        value NUMERIC
    );
    
    CREATE TABLE vbe_pca (
        dao_id VARCHAR(255),
        vbe_window INTEGER,
        pca_x NUMERIC,
        pca_y NUMERIC,
        label INTEGER
    );

    CREATE TABLE vbe_dao (
        dao_id VARCHAR(255),
        vbe_window INTEGER,
        vbe NUMERIC,
        vbe_min_entropy NUMERIC,
        cluster_0 INTEGER,
        cluster_1 INTEGER,
        cluster_2 INTEGER,
        cluster_0_pct INTEGER,
        cluster_1_pct INTEGER,
        cluster_2_pct INTEGER
    );

    CREATE TABLE cluster_weight (
        dao_id VARCHAR(255),
        vbe_window INTEGER,
        cluster INTEGER,
        proposal_id VARCHAR(255),
        proposal_title TEXT,
        weights NUMERIC
    );
    """
    # Execute the SQL statement
    cur.execute(create_table_sql)
    print("Tables created successfully")

def print_records(cur, table_name):
    query = "SELECT * FROM " + table_name + ";"
    cur.execute(query)
    records = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    print(colnames)
    for record in records:
        print(record)

def view_last(cur, table_name):
    query = "SELECT * FROM " + table_name + ";"
    cur.execute(query)
    colnames = [desc[0] for desc in cur.description]
    votes = cur.fetchall()
    print(colnames)
    print("Last record:", votes[-1])

def update_records(cur, table_name, field, new_value):
    query = f"UPDATE {table_name} SET {field} = %s WHERE {field} IS NULL;"
    cur.execute(query, (new_value,))
    print("Updated records successfully")

def drop_table(cur, table_name):
    query = "DROP TABLE " + table_name + ";"
    cur.execute(query)
    print("Table dropped successfully")

def drop_records(cur, table_name, field, value):
    # Edit the SQL statement below to delete records
    query = """DELETE FROM """ + table_name + """ WHERE """ + table_name + "." + field + """ = '""" + value + """';"""
    cur.execute(query)
    print("Deleted column successfully")

def view_all_tab_cols(cur):
    query="""
    SELECT 
    c.table_name, 
    c.column_name, 
    c.data_type, 
    c.is_nullable,
    tc.constraint_type
    FROM 
        information_schema.columns c
    LEFT JOIN 
        information_schema.constraint_column_usage ccu 
    ON 
        c.table_name = ccu.table_name AND c.column_name = ccu.column_name
    LEFT JOIN 
        information_schema.table_constraints tc 
    ON 
        ccu.constraint_name = tc.constraint_name
    WHERE 
        c.table_schema = 'public'
    ORDER BY 
        c.table_name, c.ordinal_position;
    """
    cur.execute(query)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    print(colnames)
    # Print each row
    for row in rows:
        print(row)

def find_protocol(dao_id, dao_name):
    if dao_id in ['2206072049871356990', 'opcollective.eth']:
        return "Optimism"
    elif dao_id in ['2206072050458560434', 'uniswapgovernance.eth']:
        return "Uniswap"
    elif dao_id in ['2206072050315953936', 'arbitrumfoundation.eth']:
        return "Arbitrum"
    elif dao_id in ['2206072050458560426', 'ens.eth']:
        return "ENS"
    else:
        return dao_name

# Main function
if __name__ == "__main__":
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute("SET search_path TO public;")

        # Uncomment and use functions below as needed
        view_all_tab_cols(cur) # View all columns in the tables
        # create_tables(cur) # Should only be run once
        # update_records(cur, 'votes', 'platform', 'Snapshot') # Update records
        # view_last(cur, 'vbe_dao') # View the last record in the table
        # print_records(cur, 'vbe_dao') # Print all records in the table

        # Caution while using the functions below
        # drop_table(cur, 'cluster_weight') # Drop the table
        # drop_records(cur, 'vbe_dao', 'dao_id', 'gnosis.eth') # Delete records
        
        conn.commit()

    except (Exception, psycopg2.Error) as error:
        print(f"Error: {error}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("Database connection closed.")