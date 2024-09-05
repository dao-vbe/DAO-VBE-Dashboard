"""
tally_api.py

This module provides functionality to interact with the Tally API for retrieving on-chain DAO (Decentralized Autonomous Organization) data.

The TallyAPI class encapsulates methods for fetching information about DAOs, proposals, and voting data from the Tally platform.

Key features:
- Fetching DAO information
- Retrieving proposal details
- Collecting voting data for proposals
- Creating structured DataFrames from the fetched data

Note: Ensure that the required environment variables (TALLY_API_URL, TALLY_API_KEY) are set before using this module.
"""

import csv
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import database as db
import requests
import time
import pandas as pd
from datetime import datetime
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class TallyAPI:
    def __init__(self):
        """
        Initializes the TallyAPI class with the necessary API URL and headers.
        """
        self.tally_api_url = os.getenv('TALLY_API_URL')
        self.tally_headers = {
            "Api-key": os.getenv('TALLY_API_KEY'),
            "Content-Type": "application/json"
        }
        self.organization_query = """
        query Organization($input: OrganizationInput!) {
          organization(input: $input) {
            id
            slug
            name
            metadata {
              description
            }
            creator {
              address
              name
            }
            proposalsCount
            delegatesCount
            delegatesVotesCount
          }
        }
        """
        self.proposal_query = """
        query Proposals($input: ProposalsInput!) {
          proposals(input: $input) {
            nodes {
              ... on Proposal {
                id
                onchainId
                status
                quorum
                createdAt
                start {
                  ... on Block {
                    timestamp
                  }
                }
                end {
                  ... on Block {
                    timestamp
                  }
                }
                metadata {
                  title
                  description
                  eta
                }
                creator {
                  address
                }
                proposer {
                  address
                }
                voteStats {
                  type
                  votesCount
                  votersCount
                  percent
                }
              }
            }
            pageInfo {
              firstCursor
              lastCursor
              count
            }
          }
        }
        """
        self.votes_query = """
        query Votes($input: VotesInput!) {
          votes(input: $input) {
            nodes {
              ... on Vote {
                voteId: id
                amount
                reason
                type
                voter {
                  address
                  name
                  type
                }
              }
            }
            pageInfo {
              firstCursor
              lastCursor
              count
            }
          }
        }
        """

    def fetch_daos(self, dao_slugs):
        """
        Fetches DAO data from the Tally API based on the provided slugs.

        Args:
            dao_slugs (list): A list of slugs for the DAOs to be fetched.

        Returns:
            list: A list of dictionaries containing the DAO data.
        """
        all_daos = []
        for slug in dao_slugs:
            variables = {"input": {"slug": slug}}

            try:
                response = requests.post(
                    self.tally_api_url,
                    headers=self.tally_headers,
                    json={"query": self.organization_query, "variables": variables}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    dao = data['data']['organization']
                    if dao:
                        all_daos.append(dao)
                    else:
                        print(f"No data found for DAO with slug: {slug}")
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"Rate limited. Waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    # Retry this slug
                    dao = self.fetch_daos([slug])
                    if dao:
                        all_daos.extend(dao)
                else:
                    print(f"Query failed for {slug} with status code {response.status_code}")
                    print("Response content:", response.content.decode())
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {slug}:", e)

        return all_daos

    def fetch_proposals(self, org_ids):
        """
        Fetches proposal data from the Tally API based on the provided organization IDs.

        Args:
            org_ids (list): A list of organization IDs for the DAOs to be fetched.

        Returns:
            list: A list of dictionaries containing the proposal data.
        """
        all_proposals = []
        for org_id in org_ids:
            proposal_variables = {
                "input": {
                    "filters": {
                        "organizationId": org_id,
                        "includeArchived": True,
                    },
                    "page": {},
                    "sort": {
                        "isDescending": False,
                        "sortBy": "id"
                    }
                }
            }

            while True:
                try:
                    response = requests.post(
                        self.tally_api_url,
                        headers=self.tally_headers,
                        json={"query": self.proposal_query, "variables": proposal_variables}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'errors' in data:
                            print(f"Query failed for organization ID {org_id}:")
                            for error in data['errors']:
                                print(error['message'])
                            break
                        
                        proposals = data.get('data', {}).get('proposals', {}).get('nodes', [])
                        for proposal in proposals:
                            proposal['organizationId'] = org_id
                        all_proposals.extend(proposals)
                        
                        page_info = data['data']['proposals']['pageInfo']
                        last_cursor = page_info.get('lastCursor')
                        
                        if not last_cursor:
                            break
                        
                        proposal_variables['input']['page']['afterCursor'] = last_cursor
                    
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        print(f"Rate limited. Waiting for {retry_after} seconds...")
                        time.sleep(retry_after)
                    
                    else:
                        print(f"Query failed for organization ID {org_id} with status code {response.status_code}")
                        print("Response content:", response.content.decode())
                        break
                
                except requests.exceptions.RequestException as e:
                    print(f"Request failed for organization ID {org_id}:", e)
                    break
        
        return all_proposals

    def fetch_voting_data(self, all_proposals):
        """
        Fetches voting data for the provided proposal IDs.

        Args:
            all_proposals (list): A list of proposal IDs for which voting data is to be fetched.

        Returns:
            list: A list of tuples, where each tuple contains the proposal ID and the corresponding voting data.
        """
        all_votes = []
        for proposal_id in all_proposals:
            print(f"Fetching votes for proposal ID: {proposal_id}")
            vote_variables = {
                "input": {
                    "filters": {
                        "proposalId": proposal_id,
                        "includePendingVotes": False
                    },
                    "page": {},
                    "sort": {
                        "isDescending": False,
                        "sortBy": "id"
                    }
                },
            }
            
            proposal_votes = []
            while True:
                try:
                    response = requests.post(
                        self.tally_api_url,
                        headers=self.tally_headers,
                        json={"query": self.votes_query, "variables": vote_variables}
                    )
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except json.JSONDecodeError as e:
                            print("Failed to decode JSON:", e)
                            print("Response content:", response.content.decode())
                            break
                        
                        votes = data.get('data', {}).get('votes', {}).get('nodes', [])
                        proposal_votes.extend(votes)
                        
                        page_info = data['data']['votes']['pageInfo']
                        last_cursor = page_info.get('lastCursor')
                        
                        if not last_cursor:
                            break
                        
                        vote_variables['input']['page']['afterCursor'] = last_cursor
                    
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 3))  
                        print(f"Rate limited. Waiting for {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue 
                    
                    else:
                        print(f"Query failed with status code {response.status_code}")
                        print("Response content:", response.content.decode())
                        break
                
                except requests.exceptions.RequestException as e:
                    print("Request failed:", e)
                    break
            
            # Process and append the votes for this proposal to the CSV file
            all_votes.append((proposal_id, proposal_votes))

        return all_votes

    @staticmethod
    def load_json_data(file):
        """
        Loads JSON data from a file.

        Args:
            file (str): The path to the JSON file.

        Returns:
            dict: The JSON data loaded from the file.
        """
        with open(file, "r") as file:
            file_data = json.load(file)

        return file_data

    @staticmethod
    def clean_text(text):
        """
        Cleans the provided text by removing newlines, carriage returns, and commas.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        cleaned_text = text.replace('\n', ' ').replace('\r', ' ')
        cleaned_text = cleaned_text.replace(',', ';')
        cleaned_text = cleaned_text.strip()
        if len(cleaned_text) > 32000:
            cleaned_text = cleaned_text[:32000]

        return cleaned_text

    @staticmethod
    def create_daos_df(dao_data):
        """
        Creates a DataFrame for DAOs based on the provided DAO data.

        Args:
            dao_data (list): A list of dictionaries containing the DAO data.

        Returns:
            DataFrame: A DataFrame containing the DAO data.
        """
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
            
        daos_list = []
        for dao in dao_data:
            dao_id = dao['id']
            dao_name = dao['name']
            dao_slug = dao['slug']
            description = dao['metadata'].get('description')
            protocol = find_protocol(dao['id'], dao['name'])
            member_count = dao.get('delegatesCount')
            proposal_count = dao.get('proposalsCount')
            active_voters = None
            votes_cast = dao.get('delegatesVotesCount')

            daos_list.append({
                'platform': 'Tally',
                'dao_id': dao_id,
                'dao_name': dao_name,
                'dao_slug': dao_slug,
                'about': description,
                'protocol': protocol,
                'member_count': member_count,
                'proposal_count': proposal_count,
                'active_voters': active_voters,
                'votes_cast': votes_cast
            })

        return pd.DataFrame(daos_list)

    @staticmethod
    def create_proposals_df(proposals_data, dao_df):
        """
        Creates a DataFrame for proposals based on the provided proposals data and DAO data.

        Args:
            proposals_data (list): A list of dictionaries containing the proposals data.
            dao_df (DataFrame): A DataFrame containing the DAO data.

        Returns:
            DataFrame: A DataFrame containing the proposals data.
        """
        proposals_list = []
        dao_dict = dao_df.set_index('dao_id')['dao_name'].to_dict()
        for proposal in proposals_data:
            dao_id = proposal['organizationId']
            dao_name = dao_dict.get(dao_id, "Not Found")
            proposal_id = proposal['id']
            proposal_title = proposal['metadata'].get('title')
            proposal_body = proposal['metadata'].get('description')

            # Clean and truncate the proposal body
            clean_body = TallyAPI.clean_text(proposal_body)

            choices = [stat['type'] for stat in proposal.get('voteStats', [])]
            choice_scores = [stat['votesCount'] for stat in proposal.get('voteStats', [])]
            start_date = proposal['start'].get('timestamp') if proposal.get('start') else None
            end_date = proposal['end'].get('timestamp') if proposal.get('end') else None
            created = proposal.get('createdAt')
            state = proposal.get('status')
            quorum = proposal.get('quorum')
            creator = proposal.get('creator', {}).get('address')
            proposer = proposal.get('proposer', {}).get('address')

            proposals_list.append({
                'platform': 'Tally',
                'dao_id': dao_id,
                'proposal_id': proposal_id,
                'proposal_title': proposal_title,
                'proposal_body': clean_body,  
                'choices': choices,
                'start_date': start_date,
                'end_date': end_date,
                'created': created,
                'state': state,
                'choice_scores': choice_scores,
                'quorum': quorum,
                'creator': creator,
                'proposer': proposer
            })

        return pd.DataFrame(proposals_list)

    @staticmethod
    def create_voters_df(voting_data):
        """
        Creates a DataFrame for voters based on the provided voting data.

        Args:
            voting_data (list): A list of tuples, where each tuple contains the proposal ID and the corresponding voting data.

        Returns:
            DataFrame: A DataFrame containing the voters data.
        """
        voters_list = []
        for proposal_id, votes in voting_data:
            for vote in votes:
                vote_id = vote['voteId']
                voter_address = vote['voter']['address']
                voter_name = vote['voter'].get('name')
                choice = vote.get('type')
                voting_power = vote.get('amount')
                reason = vote.get('reason')
                discussion = None 

                voters_list.append({
                    'platform': 'Tally',
                    'proposal_id': proposal_id,
                    'vote_id': vote_id,
                    'voter_address': voter_address,
                    'voter_name': voter_name,
                    'choice': choice,
                    'voting_power': voting_power,
                    'reason': reason,
                    'discussion': discussion
                })

        return pd.DataFrame(voters_list)

def main():
    # Attempt connection to DB
    sql_handler = db.DatabaseHandler()

    voter_db_df = sql_handler.db_to_df('votes')
    proposal_db_df = sql_handler.db_to_df('proposals')
    dao_db_df = sql_handler.db_to_df('dao')
    
    dao_input_df = pd.read_csv("data_input/dao_input.csv").query('platform == "Tally"')
    dao_slugs = dao_input_df['dao_slug'].tolist()
    dao_ids = dao_input_df['dao_id'].tolist()
    
    print("Fetching DAOs...")
    tally_api = TallyAPI()
    dao_df = tally_api.create_daos_df(tally_api.fetch_daos(dao_slugs))

    dao_df_new = dao_df[~dao_df['dao_id'].isin(dao_db_df['dao_id'])]
    if not dao_df_new.empty:
        print("Writing new DAOs to SQL", dao_df_new['dao_name'].unique())
        sql_handler.df_to_sql(dao_df_new, 'dao', 'append')

    print("Fetching proposals...")
    all_proposals = tally_api.fetch_proposals(dao_ids)
    proposal_df = tally_api.create_proposals_df(all_proposals, dao_df)

    # If proposal id is not in the database, write to SQL
    unseen_proposals = proposal_df[~proposal_df['proposal_id'].isin(proposal_db_df['proposal_id'])]
    if not unseen_proposals.empty:
        print("Writing new proposals to SQL for", unseen_proposals['dao_name'].unique())
        sql_handler.df_to_sql(unseen_proposals, 'proposals', 'append')

    if input("Do you want to fetch voting data? (yes/no): ").strip().lower() == 'yes':
        proposal_df = proposal_df.merge(dao_df[['dao_name', 'dao_id']], on='dao_name', how='left')
        for dao_id in dao_df['dao_id'].unique():
            fetch_proposal_list = proposal_df[proposal_df['dao_id'] == dao_id]['proposal_id'].tolist()
            voting_data = tally_api.fetch_voting_data(fetch_proposal_list)

            # If proposal id is not in the votes table, write to SQL
            voters_df = tally_api.create_voters_df(voting_data)
            new_votes = voters_df[~voters_df['proposal_id'].isin(voter_db_df['proposal_id'])]
            if not new_votes.empty:
                print("Writing new votes to SQL for", new_votes['proposal_id'].unique())
                sql_handler.df_to_sql(new_votes, 'votes', 'append')

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
