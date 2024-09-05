"""
This module performs clustering analysis on DAO proposal, voting data, and category clusters.
"""

import ast
import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import database as db

class DataProcessor:
    @staticmethod
    def process_dao(dao, proposal_df, voter_df):
        mask = proposal_df['dao_id'] == dao
        dao_df = proposal_df[mask]
        return dao_df.drop_duplicates()
    
    @staticmethod
    def merge_data(window_df, voter_df):
        if len(window_df['proposal_id'].unique()) != 10:
            print("Stopping, proposal window smaller than 10")
            return None

        proposal_id_list = window_df['proposal_id'].unique()
        # print("Proposal ID List: ", proposal_id_list)
        mask = voter_df['proposal_id'].isin(proposal_id_list)
        voter_df = voter_df[mask]
        if len(voter_df) == 0:
            print("Stopping, no votes for proposals in window", proposal_id_list)
            return None
        voter_df.to_csv('data_output/sample_choice_DELETE.csv', index=False)
        
        # Make sure all voters are represented even if they don't vote on a proposal
        unique_voter_proposals = pd.MultiIndex.from_product([voter_df['voter_address'].unique(), window_df['proposal_id'].unique()], names=['voter_address', 'proposal_id']).to_frame(index=False)
        new_voter_df = pd.merge(unique_voter_proposals, voter_df, how='left', on=['voter_address', 'proposal_id'])
        new_voter_df['choice_position'] = new_voter_df['choice_position'].fillna(0)
        new_voter_df['voting_power'] = pd.to_numeric(new_voter_df['voting_power'], errors='coerce')
        new_voter_df.to_csv('data_output/sample_choice_DELETE.csv', index=False)

        grouped = new_voter_df.groupby('voter_address')['voting_power']

        def fill_voting_power(series):
            mean_value = series.mean(skipna=True)  
            return series.fillna(mean_value)
        
        transformed_voting_power = grouped.transform(fill_voting_power)
        new_voter_df['voting_power'] = transformed_voting_power
        new_voter_df['voting_power'] = new_voter_df.groupby('voter_address')['voting_power'].transform(lambda x: x.fillna(x.mean()))

        training_df = pd.pivot_table(new_voter_df, values=['choice_position'], index=['voter_address', 'voting_power'], columns=['proposal_id'], fill_value=0)
        training_df.reset_index(inplace=True)
        training_df.columns = training_df.columns.get_level_values(1)
        training_df.to_csv('data_output/sample_choice_DELETE2.csv', index=False)

        training_df.columns = ['voter_address', 'voting_power'] + list(training_df.columns[2:])

        return training_df

class Clusterer:
    @staticmethod
    def cluster_data(training_df, X_train, scaled_data):
        # Implement logic to find optimal number of clusters
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        if len(scaled_data) < 3:
            print(f"Skipping clustering: not enough samples ({len(scaled_data)}) for 3 clusters.")
            return None, None, None, None
        dao_kmeans = model.fit(scaled_data)
        labels_count = np.bincount(dao_kmeans.labels_)

        for label, count in enumerate(labels_count):
            pct = (count / sum(labels_count)) * 100
            # print(f"Cluster {label}: {count} occurrences, {pct:.2f}%")

        cluster_centroids = dao_kmeans.cluster_centers_
        results_df = X_train.copy()
        results_df['cluster'] = dao_kmeans.labels_

        return results_df, cluster_centroids, labels_count, model

    @staticmethod
    def graph_pca(X_train, cluster_labels, scaled_data):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(scaled_data)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        feature_names = X_train.columns
        # for i, (pc1_loading, pc2_loading) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
            # print(f"Feature {feature_names[i]}:")
            # print(f"PC1 Loading = {pc1_loading:.4f}, PC2 Loading = {pc2_loading:.4f}")

        sample_size = int(0.003 * X_pca.shape[0])
        sampled_indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
        X_pca_sampled = X_pca[sampled_indices]

        return X_pca_sampled, sampled_indices

class DataSaver:
    def __init__(self, engine):
        self.engine = engine
    def save_dao_vbe(self, dao, window_df, labels_count, window, dao_df):
        # Calculate min entropy
        if len(labels_count) < 3:
            print(f"Skipping saving for this window: only {len(labels_count)} clusters found.")
        else:
            cluster_counts = [labels_count[i] for i in range(3)]
        vbe = [max(labels_count)/sum(labels_count)]
        # cluster_counts = [labels_count[i] for i in range(3)]
        cluster_pcts = [count/sum(labels_count) for count in cluster_counts]
        vbe_value = - np.log2(max(cluster_pcts)) if max(cluster_pcts) > 0 else 0

        vbe_dao = pd.DataFrame({
            'dao_id': [dao],
            'vbe_window': [window],
            'vbe': vbe,
            'vbe_min_entropy': [vbe_value],
            'cluster_0': [cluster_counts[0]],
            'cluster_1': [cluster_counts[1]],
            'cluster_2': [cluster_counts[2]],
            'cluster_0_pct': [cluster_pcts[0]],
            'cluster_1_pct': [cluster_pcts[1]],
            'cluster_2_pct': [cluster_pcts[2]],
        })
        # Save to SQL, replace table if it's the first time, else append
        # if not self.tables_created['vbe_dao']:
        #     vbe_dao.to_sql('vbe_dao', self.engine, schema='public', if_exists='replace', index=False)
        #     self.tables_created['vbe_dao'] = True
        # else:
        #     vbe_dao.to_sql('vbe_dao', self.engine, schema='public', if_exists='append', index=False)
        if not os.path.exists('data_output/vbe_dao.csv'):
            vbe_dao.to_csv('data_output/vbe_dao.csv', index=False)
        else:
            vbe_dao.to_csv('data_output/vbe_dao.csv', mode='a', header=False, index=False)
        # vbe_dao.to_sql('vbe_dao', self.engine, schema='public', if_exists='append', index=False)

    def save_pca(self, X_train, cluster_labels, X_pca_sampled, window, dao_df, model, sampled_indices):
        PCA_X, PCA_Y = [], []
        label = model.fit_predict(X_train)[sampled_indices]

        for coord in X_pca_sampled:
            x, y = coord[0], coord[1]
            PCA_X.append(x)
            PCA_Y.append(y)

        dao_id = [dao_df['dao_id'].iloc[0]] * len(PCA_X)
        windows = [window] * len(PCA_X)

        vbe_pca = pd.DataFrame({
            'dao_id': dao_id,
            'vbe_window': windows,
            'pca_x': PCA_X,
            'pca_y': PCA_Y,
            'label': label,
        })
        
        # Save to SQL, replace table if it's the first time, else append
        # if not self.tables_created['vbe_pca']:
        #     vbe_pca.to_sql('vbe_pca', self.engine, schema='public', if_exists='replace', index=False)
        #     self.tables_created['vbe_pca'] = True
        # else:
        #     vbe_pca.to_sql('vbe_pca', self.engine, schema='public', if_exists='append', index=False)
        if not os.path.exists('data_output/vbe_pca.csv'):
            vbe_pca.to_csv('data_output/vbe_pca.csv', index=False)
        else:
            vbe_pca.to_csv('data_output/vbe_pca.csv', mode='a', header=False, index=False)
        # vbe_pca.to_sql('vbe_pca', self.engine, schema='public', if_exists='append', index=False)

    def save_cluster_weights(self, cluster_centroids, window_df, window, dao):
        data = []
        for cluster_index, cluster_data in enumerate(cluster_centroids):
            if len(window_df) == 10:
                for proposal_index, (_, row) in enumerate(window_df.iterrows()):
                    data.append({
                        'dao_id': dao,
                        'vbe_window': window,
                        'cluster': cluster_index,
                        'proposal_id': row['proposal_id'],
                        'proposal_title': row['proposal_title'],
                        'weights': cluster_data[proposal_index]
                    })

        cluster_explain = pd.DataFrame(data)
        # if not self.tables_created['cluster_weight']:
        #     cluster_explain.to_sql('cluster_weight', self.engine, schema='public', if_exists='replace', index=False)
        #     self.tables_created['cluster_weight'] = True
        # else:
        #     cluster_explain.to_sql('cluster_weight', self.engine, schema='public', if_exists='append', index=False)
        if not os.path.exists('data_output/cluster_weight.csv'):
            cluster_explain.to_csv('data_output/cluster_weight.csv', index=False)
        else:
            cluster_explain.to_csv('data_output/cluster_weight.csv', mode='a', header=False, index=False)
        # cluster_explain.to_sql('cluster_weight', self.engine, schema='public', if_exists='append', index=False)

class MainProcessor:
    def __init__(self):
        self.db_connector = db.DatabaseHandler()
        self.data_processor = DataProcessor()
        self.clusterer = Clusterer()
        self.data_saver = DataSaver(self.db_connector.engine)

    def process_proposals_in_windows(self, dao, dao_df, voter_df):
        dao_df_sorted = dao_df.sort_values(by='end_date', ascending=False)
        # Remove if status "active", "pending", "queued"
        states_to_exclude = ['active', 'queued', 'pending']
        dao_df_sorted = dao_df_sorted[~dao_df_sorted['state'].isin(states_to_exclude)]
        total_rows = len(dao_df_sorted)
        end = total_rows - 9
        windows = 0

        for start in range(end):
            window = end - start
            if window == 121 and dao == 'gnosis.eth':
                continue
            print("Processing window #: ", window)

            window_df = dao_df_sorted.iloc[start:start + 10]
            training_df = self.data_processor.merge_data(window_df, voter_df)
            if training_df is None:
                continue
            X_train = training_df[training_df.columns[2:]]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X_train)

            # print("Clustering data")
            results_df, cluster_centroids, cluster_labels, model = self.clusterer.cluster_data(training_df, X_train, scaled_data)
            if results_df is None:
                continue
            X_pca_sampled, sampled_indices = self.clusterer.graph_pca(X_train, cluster_labels, scaled_data)

            self.data_saver.save_dao_vbe(dao, window_df, cluster_labels, window, dao_df)
            self.data_saver.save_pca(X_train, cluster_labels, X_pca_sampled, window, dao_df, model, sampled_indices)
            self.data_saver.save_cluster_weights(cluster_centroids, window_df, window, dao)

            windows += 1

        print(f"Total windows processed: {windows}")

    def run(self):
        proposal_df = self.db_connector.db_to_df("proposals")
        voter_df = self.db_connector.db_to_df("votes")
        # vbe_dao_df = pd.read_csv('data_output/vbe_dao.csv')
        # vbe_dao_existing = self.db_connector.db_to_df("vbe_dao")
        
        # Load voter data from csv for now
        # voter_df = pd.read_csv('data_output/votes_tally.csv', low_memory=False)
        # proposal_df = pd.read_csv('data_output/proposals.csv')
        # daos_df = pd.read_csv('data_output/daos.csv')

        # print("Unique values for choices", proposal_df['choices'].unique())
        voter_df['proposal_id'] = voter_df['proposal_id'].astype(str)
        proposal_df['proposal_id'] = proposal_df['proposal_id'].astype(str)

        # Voter DF add choices manually - for each proposal in voter df, get all unique values for "choice" in a list
        choices_per_proposal = voter_df.groupby('proposal_id')['choice'].unique().reset_index()
        choices_per_proposal.rename(columns={'choice': 'unique_choices'}, inplace=True)
        # Merge the result back into the original voter_df
        voter_df = voter_df.merge(choices_per_proposal, on='proposal_id', how='left')

        voter_df['choice_position'] = voter_df.apply(
            lambda row: row['unique_choices'].tolist().index(row['choice']) + 1, axis=1
        )
        
        # Check to see which dao_id are already in the vbe_dao table. If in table, skip
        
        dao_list = proposal_df['dao_id'].unique()
        
        for dao in dao_list:
            # if dao in vbe_dao_existing['dao_id'].values:
            #     continue
            print("Processing DAO", dao)
            # Returns proposals in the dao listed
            prop_df = self.data_processor.process_dao(dao, proposal_df, voter_df)
            # if dao_id contains string ".eth", continue
            if pd.isna(dao):
                continue
            self.process_proposals_in_windows(dao, prop_df, voter_df)
    
    def save_to_db(self):
        cluster_explain = pd.read_csv('data_output/cluster_weight.csv')
        cluster_explain.to_sql('cluster_weight', self.db_connector.engine, schema='public', if_exists='replace', index=False)

        vbe_dao = pd.read_csv('data_output/vbe_dao.csv')
        vbe_dao.to_sql('vbe_dao', self.db_connector.engine, schema='public', if_exists='replace', index=False)

        vbe_pca = pd.read_csv('data_output/vbe_pca.csv')
        vbe_pca.to_sql('vbe_pca', self.db_connector.engine, schema='public', if_exists='replace', index=False)
    
    def auxilliary(self):
        # save dao input and replace table in DB
        dao_df = pd.read_csv('data_input/dao_input.csv')
        dao_df.to_sql('dao_input', self.db_connector.engine, schema='public', if_exists='replace', index=False)
        # read dao_input from DB
        # dao_df = self.db_connector.db_to_df("dao_input")
        # print(dao_df['dao_id'].unique())

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    processor = MainProcessor()
    # processor.run()
    processor.auxilliary()
    # processor.save_to_db()
    print("All VBE Completed, CSVs updated")    