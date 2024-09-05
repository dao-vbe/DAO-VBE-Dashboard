"""
This module provides a class for interacting with the Boardroom API and handling related data operations.

The BoardroomAPI class offers methods to retrieve and process data from the Boardroom API,
specifically for DAO forums. It includes functionality
to fetch forum data, protocol data, and standardize URLs.

Note: This module requires the proper configuration of environment variables for Boardroom API access.
"""

import requests
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

load_dotenv()
BOARDROOM_API_KEY = os.getenv("BOARDROOM_API_KEY")
print(BOARDROOM_API_KEY)

class BoardroomAPI:
    """
    A class to interact with the Boardroom API and handle related data operations.

    This class provides methods to retrieve data from the Boardroom API, process it,
    and perform various operations DAO forums.

    Attributes:
        key (str): The API key for accessing the Boardroom API.
        forum_url_map (dict): A mapping of DAO protocols to their forum URLs.
    """

    def __init__(self) -> None:
        """
        Initializes the BoardroomAPI instance with the API key and forum URL mappings.
        """
        self.key = BOARDROOM_API_KEY
        self.forum_url_map = {
            "optimism": "https://gov.optimism.io/t/",
            "arbitrum": "https://forum.arbitrum.foundation/t/",
            "uniswap": "https://gov.uniswap.org/t/",
            "nounsdao": "https://discourse.nouns.wtf/t/"
        }

    def get_boardroom_data(self, endpoint, key, params, pages=5):
        """
        Retrieves data from the Boardroom API and returns it as a pandas DataFrame.

        Args:
            endpoint (str): The API endpoint to query.
            key (str): The API key for authentication.
            params (dict): Query parameters for the API request.
            pages (int, optional): Number of pages to retrieve. Defaults to 5.

        Returns:
            pandas.DataFrame: A DataFrame containing the data retrieved from the Boardroom API.
        """
        boardroom_url = f"https://api.boardroom.info/v1/{endpoint}?key={key}"

        next_cursor = None
        df = None

        try:
            for i in range(pages):
                if (next_cursor):
                    params['cursor'] = next_cursor
                response = requests.get(boardroom_url, params=params)
                if response.status_code == 200:
                    json_data = response.json()
                    data = json_data['data']
                    next_cursor = json_data['nextCursor'] if 'nextCursor' in json_data else None
                    if i == 0:
                        df = pd.DataFrame(data)
                    else:
                        df = pd.concat([df, pd.DataFrame(data)])
                    print(f"Page {i+1} retrieved, {len(df)} records in total")
                    if not next_cursor:
                        break
                else:
                    print(f"Failed to retrieve data: {response.status_code}")
        except Exception as e:
            print(f"Failed to retrieve data: {str(e)}")
        return df
    
    def get_discourse_data(self, params, pages=5, export_csv=True):
        """
        Retrieves discourse data from the Boardroom API.

        Args:
            params (dict): Query parameters for the API request.
            pages (int, optional): Number of pages to retrieve. Defaults to 5.
            export_csv (bool, optional): Whether to export the data to a CSV file. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved discourse data.
        """
        discourse_df = self.get_boardroom_data("discourseTopics", self.key, params, pages)
        if export_csv:
            discourse_df.to_csv("local/boardroom/br_discourse_topics.csv", index=False)
        return discourse_df
    
    def get_protocol_data(self, params, pages=5, export_csv=False):
        """
        Retrieves protocol data from the Boardroom API.

        Args:
            params (dict): Query parameters for the API request.
            pages (int, optional): Number of pages to retrieve. Defaults to 5.
            export_csv (bool, optional): Whether to export the data to a CSV file. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved protocol data.
        """
        protocol_df = self.get_boardroom_data("protocols", self.key, params, pages)
        if export_csv:
            protocol_df.to_csv("local/boardroom/br_protocols.csv", index=True)
        return protocol_df
    
    def to_url(self, protocol, slug, topic_id):
        """
        Constructs a forum URL from protocol, slug, and topic ID.

        Args:
            protocol (str): The DAO protocol.
            slug (str): The forum slug.
            topic_id (str): The topic ID.

        Returns:
            str: The constructed forum URL.
        """
        return f"{self.forum_url_map[protocol]}{slug}/{topic_id}"
    
    def get_slug_topic_id(self, protocol, url):
        """
        Extracts the slug and topic ID from a given forum URL.

        Args:
            protocol (str): The DAO protocol.
            url (str): The forum URL.

        Returns:
            tuple: A tuple containing the slug and topic ID.
        """
        split_url = url.replace(self.forum_url_map[protocol], "").split("/")
        slug, topic_id = split_url[0], split_url[1]
        if '?' in topic_id:
            topic_id = topic_id.split("?")[0]
        if '#' in topic_id:
            topic_id = topic_id.split("#")[0]
        return slug, topic_id
    
    def clean_url(self, protocol, url):
        """
        Cleans and validates a forum URL.

        Args:
            protocol (str): The DAO protocol.
            url (str): The forum URL to clean.

        Returns:
            str: The cleaned URL if valid, None otherwise.
        """
        try:
            slug, topic_id = self.get_slug_topic_id(protocol, url)
            assert topic_id.isdigit()
            return self.to_url(protocol, slug, topic_id)
        except:
            return None

    def query_from_url(self, protocol, url):
        """
        Queries Boardroom forum data from a given URL.

        Args:
            protocol (str): The DAO protocol.
            url (str): The forum URL to query.

        Returns:
            pandas.DataFrame: A DataFrame containing the queried forum data.
        """
        _, topic_id = self.get_slug_topic_id(protocol, url)
        discourse_df = self.get_discourse_data({"protocol": protocol, "topicId": topic_id})
        # Add in the URL column
        discourse_df['url'] = url
        return discourse_df


if __name__ == "__main__":
    # Target DAOs: {"cnames": "uniswap,arbitrum,optimism,nounsdao"}
    api = BoardroomAPI()

    # Example scrape a single forum URL
    # results = api.query_from_url("arbitrum", "https://forum.arbitrum.foundation/t/final-gains-network-stip-addendum/23398/3?u=cattin")
    # print(results.head())
    # results.to_csv("local/arb.csv", index=False)

    results = api.query_from_url("optimism", "https://gov.optimism.io/t/review-gf-phase-1-proposal-cycle-8-ethernautdao/3800")
    print(results.head())
