"""
combine.py

This script combines and processes DAO (Decentralized Autonomous Organization) data from multiple sources,
including Snapshot proposal data, manual proposal category annotations, and forum data from the Boardroom API.
This creates the proposal_categories table.
"""

import sys
import os
import pandas as pd
import time
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '../data_extract'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database import DatabaseHandler
from boardroom_api import BoardroomAPI

class CombineDAOData:
    def __init__(self, dao_category_dfs, dao_names):
        load_dotenv()

        self.config = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        self.handler = DatabaseHandler(self.config)
        self.boardroom_api = BoardroomAPI()

        self.dao_dfs = dao_category_dfs
        self.dao_names = dao_names

        self.column_names = {
            "id": "topic_id",
            "url": "forum_url",
            "postsCount": "post_count",
            "views": "views",
            "replyCount": "reply_count",
            "likeCount": "like_count",
            "createdAt": "forum_created_at",
            "bumpedAt": "forum_bumped_at"
        }

        # Read DB DFs
        forum_query = "SELECT * FROM forums WHERE protocol='optimism' OR protocol='arbitrum' OR protocol='uniswap'"
        self.forum_df = self.handler.execute_query(forum_query)
        self.forum_df.rename(columns=self.column_names, inplace=True)

    def get_supplement_forum_data(self, dao_df, dao):
        supplement_df = pd.DataFrame()
        for url in dao_df[dao_df['url'].isnull()]['forum_url']:
            try:
                if url:
                    print(f"Querying {url}")
                    boardroom_data = self.boardroom_api.query_from_url(dao, url)
                    if not boardroom_data.empty:
                        supplement_df = pd.concat([supplement_df, boardroom_data])
            
            except Exception as e:
                print(f"Failed to query {url}: {str(e)}")
                continue
        
        supplement_df.to_csv(f"./local/{dao}_forums_supplement.csv", index=False)
        return supplement_df

    def get_forum_data(self, dao):
        # Get Snapshot DB data
        snapshot_query = f"SELECT dao_name, proposal_id, proposal_title, created, discussion FROM proposals WHERE dao_name='{self.dao_names[dao]}' ORDER BY proposal_id ASC"
        snapshot_df = self.handler.execute_query(snapshot_query)
        snapshot_df['protocol'] = dao
        # Join snapshot_df with proposal_categories on proposal_id
        dao_df = pd.merge(snapshot_df, self.dao_dfs[dao][['proposal_id', 'category_cluster']], on='proposal_id', how='left')

        # Clean the URL column
        dao_df['forum_url'] = dao_df.apply(lambda row: self.boardroom_api.clean_url(row['protocol'], row['discussion']), axis=1)
        dao_df.drop(columns=['discussion'], inplace=True)

        # Merge with forum_df on URL
        forum_columns = self.column_names.keys()
        dao_df = pd.merge(dao_df, self.forum_df[forum_columns], left_on='forum_url', right_on='url', how='left')
        supplement_df = self.get_supplement_forum_data(dao_df, dao)
        dao_df = pd.merge(dao_df, supplement_df[forum_columns], left_on='forum_url', right_on='url', how='left', suffixes=('', '_supplement'))
        for column in forum_columns:
            if column in dao_df.columns:
                dao_df[column] = dao_df[column].fillna(dao_df[f"{column}_supplement"])

        # Drop the supplemental columns
        dao_df.drop(columns=[f"{col}_supplement" for col in forum_columns], inplace=True)
        print(dao_df)
        return dao_df

    def get_combined_dao_data(self):
        combined_df = pd.DataFrame()
        for dao in self.dao_names:
            dao_df = self.get_forum_data(dao)
            combined_df = pd.concat([combined_df, dao_df])
            
        # Write to CSV
        combined_df.to_csv("./local/combined_dao.csv", index=False)

if __name__ == "__main__":
    # We load in the categories from a manual annotation process for Uniswap, Arbitrum, and Optimism
    uni_anno_df = pd.read_csv("./local/categories/uni_categories.csv").dropna(how='all')
    arb_anno_df = pd.read_csv("./local/categories/arb_categories.csv").dropna(how='all')
    op_anno_df = pd.read_csv("./local/categories/op_categories.csv").dropna(how='all')
   
    dao_categories_dfs = {
        "optimism": op_anno_df,
        "arbitrum": arb_anno_df,
        "uniswap": uni_anno_df
    }

    dao_names = {
            "optimism": "opcollective.eth",
            "arbitrum": "arbitrumfoundation.eth",
            "uniswap": "uniswapgovernance.eth"
        }
    combiner = CombineDAOData(dao_categories_dfs, dao_names)
    combiner.get_combined_dao_data()
