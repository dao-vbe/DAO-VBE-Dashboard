# oVBE Dashboard Code

This repository contains the code for the Observable Voting Bloc Entropy (oVBE) Dashboard, a tool for analyzing Decentralized Autonomous Organization (DAO) voting data.

## Methodology

For details on our methodology, refer to [METHODOLOGY.md](METHODOLOGY.md). Below is an outline of the key steps:

1. DAO data retrieval from open source APIs such as Snapshot, Tally, Boardroom
2. Data standardization across different APIs, including significant cleaning and cross-referencing of data
3. Manually augmenting dataset according to expert consultation, and aggregating this data in a PostgreSQL database.
4. Data filtering to identify valid data for clustering and analysis, such as only counting voters that have voted on at least one proposal.
5. Calculating VBE across different organizations and within a signle organization.
6. Visualizing results.


## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables (see Configuration section)

## Configuration

Create a `.env` file in the root directory with the following variables:

```
DB_HOST=<DB_HOST>
DB_PORT=<DB_PORT>
DB_NAME=<DB_NAME>
DB_USER=<DB_USER>
DB_PASSWORD=<DB_PASSWORD>

TALLY_API_URL=https://api.tally.xyz/query
TALLY_API_KEY=<YOUR-API-KEY>

SNAPSHOT_GRAPHQL_ENDPOINT=https://hub.snapshot.org/graphql
SNAPSHOT_API_KEY=<YOUR-API-KEY>
BOARDROOM_API_KEY=<YOUR-API-KEY>
```

## Usage

1. Data Extraction:
   ```
   python data_extract/tally_api.py
   python data_extract/snapshot_api.py
   python data_extract/boardroom_api.py
   ```

2. Cluster Analysis and Data Processing:
   ```
   python cluster_analysis/combine.py
   python cluster_analysis/clustering.py
   python cluster_analysis/cluster_categories.py
   ```

3. Analytics Generation:
   ```
   python cluster_analysis/data_analytics.py
   ```

## License

This project is licensed under the MIT License.
