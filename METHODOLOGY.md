# Voting Bloc Entropy - Methodology

## Step 1: DAO Data Retrieval

We selected DAOs that fulfill at least one of the following conditions: (1) they contain 5000+ unique voters, (2) have 25+ proposals, or (3) or are one of the 20 largest treasury values according to DeepDAO. We used three different data sources: Tally for on-chain data, Snapshot for off-chain data, and Boardroom for forum metadata. We chose these sources for their quantity and quality of stored voter data, free API usage, and standardization and reproducibility for calculating VBE.

## Step 2: Data Standardization

We gathered DAO, proposal, and vote-level data from these sources, as well as open-source forum metadata. We then engaged in a significant amount of data processing, standardization, and cross-referencing between the sources, as data from the three APIs followed three different conventions. We thus use this as an opportunity to call for practitioners and data providers in the DAO space to improve standardization and interoperability of data across various platforms.

Some standardization issues include:
- The APIs follow different naming conventions for DAOs. Snapshot uses ENS names (eg. uniswapgovernance.eth), whereas Tally and Boadroom use common names (eg. Uniswap)
- Proposal IDs follow very different conventions in Snapshot and Tally, and Boardroom uses “refIDs” instead of proposal IDs.
- Proposal titles and proposal bodies on Tally use Markdown format, whereas they use plain text on Snapshot.
- Boardroom only supports forum queries by Topic ID, so we had to manually parse forum URLs into Boardroom API requests. Only some Snapshot proposals include forum URLs that we could query forum data for, and we had to manually supplement the rest.
- Voter choices are stored in an array in Snapshot, but in a dictionary for Tally, dates are timestamps in Snapshot but UNIX formatted in Tally, and Tally has fields for creator and proposers of proposals versus the single field for author in Snapshot. 

## Step 3: Manual Data Augmentation and Database Storage

After triangulating data from these sources, we also consulted with DAO community experts to manually augment the proposal data with categories of proposals, such as ecosystem funding, committee selection, and governance process amendments for three target DAOs (Uniswap, Optimism, Arbitrum) to get a qualitative sense of DAO proposal clusters. Note that for each DAO, we had slightly different category names to capture the difference in programs that each DAO runs.

We then aggregated all of the data into a PostgreSQL database hosted on AWS to create a standardized reference dataset. This database included the following tables:
- `dao_input` is the original table which incorporates manually retrieved data from DeepDAO, which can be altered to include more or less organizations in the data extraction process.
- `dao` is the output of data extraction from Tally and Snapshot, including information about their DAO, identifier, and protocol.
- `proposals` represents the unique proposals between Tally and Snapshot platforms, linked to a DAO identifier. These proposal records include proposal title, body, choices, start and end date, state, outcomes, quorum, creator address, and discussion or forum URL.
“votes” holds the proposal identifier, vote identifier, voter address, voter name (if available), choice, voting power, and reason. The next tables are transformation tables that create metrics from the original data, including dao_stats, dao_percentile, and proposal_stats. These include metrics like total votes cast, unique voters, average voting power per voter, average voter participation, Gini index, Nakamoto coefficient, and Minimum entropy. Dao_percentile stores percentile data of voting power and vote counts by voter in the organization, and proposal_stats saves information about the sum of vote counts, voting power, and average voting power for each choice in the proposal.
- `forums` holds forum metadata for proposals across select DAOs (Uniswap, Arbitrum Optimism), including views, likes, replies, posts, post created date, and post bumped date.
- `proposal_clusters` is a cleaned table that combines “proposals” with manual category annotations, and corresponding forum data for each proposal. This is available for our three target DAOs: Uniswap, Arbitrum, and Optimism.

## Step 4: Data Filtering for Valid Clustering and Analysis

Next, we cleaned up data for VBE calculations. We removed DAOs with less than 10 proposals, and for the purposes of calculating VBE, we only look at proposal statuses of “succeeded”, “failed”, and “quorum not met”. Statuses of “active”, “pending”, and “queued” data are removed. In addition, we remove proposals where voters can split their tokens amongst multiple options for data processing simplification, and also simplified ranked votes with 3 options to just have their top vote as their choice.

For voter participation, voters are only counted who have voted at least on one proposal. As an important distinction, member count is different from the measure of unique voters. In Sushigov there are over 61,000 members, but only 8,800 unique voters are counted in the voting platform data. Voter participation by taking the average of votes over total unique voters in each proposal.

## Step 5: Calculating VBE Across Different Organizations

For applications of VBE, we approached this with two angles: the first, which is VBE for inter-organizational comparison, and the second for VBE within an organization. In the VBE calculated across organizations, we use windows of 10 proposals at a time to cluster voters by their choice selection. These windows slide using proposals ordered by their end date, and clustering is applied to voters in the window uniformly using the same clustering method and hyperparameters (i.e., number of clusters, optimization method, distance method, scaling, and entropy function for calculating VBE). 

Because oVBE leverages any clustering and entropy methods, the one chosen for the data in DAO windows use K-means with a k value of 3, elbow method for optimization of clusters, cosine distance, standard scaling, and both minimum and Shannon entropy for calculating VBE. 

For calculating VBE within an organization split by different rounds, we calculate separately the off-chain voting rounds of a DAO and the on-chain voting rounds, before comparing these values.

## Step 6: Data Visualization

Finally, we use Salesforce Tableau to visualize VBE and comparative metrics in several different views. At the DAO level, for example, we visualize:
- DAO Name
- Voting platform
- Treasury size (in dollars)
- DAO category
- Membership count
- Unique voters
- Total proposal count
- Total votes cast
- Average voter participation rate
- Voting Bloc Entropy (using minimum entropy)
- Voting Bloc Entropy standard deviation
- Voting Bloc Entropy (using Shannon entropy)
- Nakamoto coefficient (Minimal actors that hold more than 50% voting power)
- Gini coefficient (Voting Power holdings)
