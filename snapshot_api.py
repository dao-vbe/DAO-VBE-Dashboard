"""
snapshot_api.py

This module provides functionality to interact with the Snapshot API for retrieving off-chain DAO (Decentralized Autonomous Organization) voting data.

The SnapshotAPI class encapsulates methods for fetching information about DAOs, proposals, and voting data from the Snapshot platform.

Key features:
- Fetching DAO information
- Retrieving proposal details
- Collecting voting data for proposals
- Creating structured DataFrames from the fetched data

Note: Ensure that the required environment variables (SNAPSHOT_API_KEY) are set before using this module.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import database as db
import requests
import time
import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class SnapshotAPI:
    """
    A class to interact with the Snapshot API and handle related data operations.

    This class provides methods to retrieve data from the Snapshot API, process it,
    and perform various operations related to DAOs and proposals.

    Attributes:
        GRAPHQL_ENDPOINT (str): The URL of the GraphQL endpoint for the Snapshot API.
        HEADERS (dict): The headers to be used in API requests.
    """
    GRAPHQL_ENDPOINT = "https://hub.snapshot.org/graphql"
    HEADERS = {'x-api-key': os.getenv('SNAPSHOT_API_KEY')}

    def __init__(self):
        """
        Initializes the SnapshotAPI class and sets up the GraphQL queries.
        """
        self.dao_query = """
        query GetDao($dao: String!) {
          space(id: $dao) {
            id
            name
            about
          }
        }
        """
        self.daos_query = """
        query GetDaos($offset: Int, $limit: Int){
            spaces(
              first: $limit,
              skip: $offset
              orderBy: "members",
              orderDirection: desc
            ) 
            {
              id
              name
              about
              members
            }
          }
        """
        self.proposals_query = """
        query GetProposals($space: String!) {
          proposals(
            first: 300,
            where: {
              space_in: [$space],
              state: "closed"
            },
            orderBy: "created",
            orderDirection: desc
          ) {
            id
            title
            body
            choices
            discussion
            start
            end
            created
            quorum
            snapshot
            state
            scores
            scores_by_strategy
            scores_total
            scores_updated
            author
          }
        }
        """
        self.votes_query = """
        query GetVotes($proposal: String!, $offset: Int, $limit: Int, $min_vp: Float) {
          votes(
            first: $limit,
            skip: $offset
            where: {
              proposal: $proposal
              vp_lte: $min_vp
            }
            orderBy: "vp",
            orderDirection: desc
          ) {
            voter
            id
            choice
            vp
            proposal {
              id
              title
              choices
            }
            reason
          }
        }
        """

    def run_query(self, query, variables):
        """
        Executes a GraphQL query and handles the response.

        Args:
            query (str): The GraphQL query to be executed.
            variables (dict): The variables to be used in the query.

        Returns:
            dict: The data returned from the API response.
        """
        # print(f"Running query: {query}, with variables: {variables}")
        request = requests.post(self.GRAPHQL_ENDPOINT, json={'query': query, 'variables': variables}, headers=self.HEADERS)
        time.sleep(0.25) # API rate limiting
        if request.status_code == 200:
            data = request.json()['data']
            return data
        else:
            raise Exception(f"Query failed with status code {request.status_code}")

    def get_all_daos(self, dao_list):
        """
        Retrieves information about all DAOs in the provided list.

        Args:
            dao_list (list): A list of DAO IDs.

        Returns:
            list: A list of dictionaries containing the DAO data.
        """
        daos_response = []
        for dao_id in dao_list:
            dao_data = self.run_query(self.dao_query, {"dao": dao_id})
            
            if dao_data and 'space' in dao_data:
                space = dao_data['space']
                daos_response.append(space)
        print(f"Processed all DAOs, Total DAOs: {len(daos_response)}")

        return daos_response

    def get_proposals(self, space):
        """
        Retrieves proposals for a given DAO.

        Args:
            space (str): The ID of the DAO.

        Returns:
            list: A list of dictionaries containing the proposal data.
        """
        return self.run_query(self.proposals_query, {'space': space})['proposals']

    def get_all_votes(self, proposal_id):
        """
        Retrieves all votes for a given proposal.

        Args:
            proposal_id (str): The ID of the proposal.

        Returns:
            list: A list of dictionaries containing the vote data.
        """
        votes_response = []
        seen_voters = set()
        min_vp = None
        skip = 0
        limit = 1000

        while True:
            votes = self.run_query(self.votes_query, {'proposal': proposal_id, 'skip': skip, 'limit': limit, 'min_vp': min_vp})['votes']
            if not votes:
                break
            
            new_votes = 0
            for vote in votes:
                voter = vote['voter']
                if voter not in seen_voters:
                    votes_response.append(vote)
                    seen_voters.add(voter)
                    new_votes += 1
            if new_votes == 0:
                break

            min_vp = votes_response[-1]['vp']
            skip += 1000
            print(f"{len(seen_voters)} voters, min vp: {min_vp}")

            if len(votes) < 1000:
                break

        return votes_response

class DataProcessor:
    """
    A class to process data related to DAOs and proposals from the Snapshot API.

    This class provides methods to create DataFrames for DAOs and proposals,
    and to process voting data.
    """
    @staticmethod
    def make_dao_table(dao_response):

        def find_protocol(dao_id, dao_name):
            if dao_id in ['2206072049871356990','opcollective.eth']:
                return "Optimism"
            elif dao_id in ['2206072050458560434','uniswapgovernance.eth']:
                return "Uniswap"
            elif dao_id in ['2206072050315953936','arbitrumfoundation.eth']:
                return "Arbitrum"
            elif dao_id in ['2206072050458560426','ens.eth']:
                return "ENS"
            else:
                return dao_name
        platform = "Snapshot"
        dao_name = [dao['name'] for dao in dao_response]
        dao_slug = [dao['id'] for dao in dao_response]
        dao_about = [dao['about'] for dao in dao_response]
        protocol = [find_protocol(dao['id'], dao['name']) for dao in dao_response] # Logic to use find_protocol based on dao_slug/dao_id
        dao_df = pd.DataFrame({'platform': platform, 'dao_id': dao_slug, 'dao_name': dao_name, 'dao_slug': dao_slug, 'about': dao_about, 'protocol': protocol})
        dao_df['dao_id'] = dao_df['dao_id'].astype(str)
        return dao_df

    @staticmethod
    def get_viable_proposals(proposals):
        """
        Filters proposals to only include those with 3 or fewer choices.

        Args:
            proposals (list): A list of dictionaries containing the proposal data.

        Returns:
            list: A list of dictionaries containing the filtered proposal data.
        """
        return [proposal for proposal in proposals if len(proposal['choices']) <= 3]

    @staticmethod
    def add_proposal_table(space, proposal_response):
        """
        Creates a DataFrame for proposals.

        Args:
            space (str): The ID of the DAO.
            proposal_response (list): A list of dictionaries containing the proposal data.

        Returns:
            pd.DataFrame: A DataFrame containing the proposal data.
        """
        def unix_to_timestamp(unix_time):
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(unix_time))
        data = {
            'platform': ['Snapshot'] * len(proposal_response),
            'dao_id': [space] * len(proposal_response),
            'proposal_id': [p['id'] for p in proposal_response],
            'proposal_title': [p['title'] for p in proposal_response],
            'proposal_body': [p['body'] for p in proposal_response],
            'choices': [p['choices'] for p in proposal_response],
            'start_date': [unix_to_timestamp(p['start']) for p in proposal_response],
            'end_date': [unix_to_timestamp(p['end']) for p in proposal_response],
            'created': [unix_to_timestamp(p['created']) for p in proposal_response],
            'state': [p['state'] for p in proposal_response],
            'choice_scores': [p['scores'] for p in proposal_response],
            'quorum': [p['quorum'] for p in proposal_response],
            'creator': [p['author'] for p in proposal_response],
            'proposer': [p['author'] for p in proposal_response],
            'discussion': [p['discussion'] for p in proposal_response]
        }
        return pd.DataFrame(data)

    @staticmethod
    def add_voter_table(votes_response):
        """
        Creates a DataFrame for votes.

        Args:
            votes_response (list): A list of dictionaries containing the vote data.

        Returns:
            pd.DataFrame: A DataFrame containing the vote data.
        """
        def get_choice(v):
            if isinstance(v['choice'], dict):
                max_key = max(v['choice'], key=v['choice'].get)
                print(max_key)
                choice = int(max_key) -1
            elif isinstance(v['choice'], list):
                print(v['choice'])
                if len(v['choice']) < 1:
                    return None
                else:
                    choice = v['choice'][0] - 1
            else:
                choice = v['choice'] - 1
            return v['proposal']['choices'][choice]
        data = {
            'platform': ['Snapshot'] * len(votes_response),
            'proposal_id': [v['proposal']['id'] for v in votes_response],
            'vote_id': [v['id'] for v in votes_response],
            'voter_address': [v['voter'] for v in votes_response],
            'choice': [get_choice(v) for v in votes_response],
            'voting_power': [v['vp'] for v in votes_response],
            'reason': [v['reason'] for v in votes_response],
        }
        return pd.DataFrame(data)

    @staticmethod
    def process_choice_column(df):
        """
        Processes the choice column to get the maximum value.

        Args:
            df (pd.DataFrame): A DataFrame containing the vote data.

        Returns:
            pd.DataFrame: A DataFrame containing the processed vote data.
        """
        def get_max_value(d):
            return d[max(d, key=d.get)] if isinstance(d, dict) else d

        df['choice'] = df['choice'].apply(lambda x: get_max_value(x) if isinstance(x, dict) else (x[0] if isinstance(x, list) else x))
        return df

def main():
    sql_handler = db.DatabaseHandler()
    data_fetcher = SnapshotAPI()
    data_processor = DataProcessor()

    voter_db_df = sql_handler.db_to_df('votes')
    proposal_db_df = sql_handler.db_to_df('proposals')
    dao_db_df = sql_handler.db_to_df('dao')

    # Read dao_input.csv as DAO List
    dao_list = pd.read_csv('data_input/dao_input.csv')
    dao_list = dao_list.query('platform == "Snapshot"')['dao_id'].tolist()

    # Process DAOs and write to SQL
    dao_response = data_fetcher.get_all_daos(dao_list)
    dao_df = data_processor.make_dao_table(dao_response)
        
    # If DAO is not in the database, write to SQL
    dao_df_new = dao_df[~dao_df['dao_id'].isin(dao_db_df['dao_id'])]

    if not dao_df_new.empty:
        print("Writing new DAOs to SQL")
        sql_handler.df_to_sql(dao_df_new, 'dao', 'append')
    # dao_df.to_csv('data_output/dao_snapshot.csv', index=False) # Write to CSV

    # Process proposals by DAO and write to SQL 
    for dao_id in dao_list:
        print("Processing ", dao_id)
        # if dao_id == "stgdao.eth":
        #     print("Skipping", dao_id)
        #     continue
        proposals = data_fetcher.get_proposals(dao_id)
        viable_proposals = [proposal for proposal in proposals if len(proposal['choices']) <= 3]
        proposal_df = data_processor.add_proposal_table(dao_id, viable_proposals)
        
        unseen_proposals = proposal_df[~proposal_df['proposal_id'].isin(proposal_db_df['proposal_id'])]
        # If proposal id is not in the database, write to SQL
        if not unseen_proposals.empty:
            print("Writing new proposals to SQL for", dao_id)
            sql_handler.df_to_sql(unseen_proposals, 'proposals', 'append')
        # proposal_df.to_csv("data_output/proposals_snapshot.csv", mode='a', header=False, index=False)
        
        unseen_votes = proposal_df[~proposal_df['proposal_id'].isin(voter_db_df['proposal_id'])]
        # Process votes in the proposal
        for proposal in unseen_votes['proposal_id']:
            if proposal == "0x44b9630efed11ff179b69646989d1ef61f05a143164773021844b6aa06878c2a": # stops at 215664 votes
                continue
            print("Processing votes for proposal", proposal)
            votes = data_fetcher.get_all_votes(proposal)
            voter_df = data_processor.add_voter_table(votes)
            voter_df = data_processor.process_choice_column(voter_df)

            sql_handler.df_to_sql(voter_df, 'votes', 'append')
            # voter_df.to_csv("data_output/votes_snapshot.csv", mode='a', header=False, index=False)

if __name__ == "__main__":
    main()