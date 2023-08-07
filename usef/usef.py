"""
	Authors: Ash, Kinga, Agata, Akshat 
	Date: Feb 2022
"""
import os
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class USEF:


	def __init__(self, config):
		self.config = config
		self.graph_size = None
		self.emb = None
		self.nf = None
		self.emb_cols = []
		self.nf_cols = []


	def run(self):
		self.load_raw_data()
		self.normalize_data()
		self.k_means_clustering()
		sample_ens = []
		for i in range(self.config["sample_ensemble"]):
			self.within_nodes, self.between_nodes = self.get_node_feature_samples()
			self.nf_w_dst, self.nf_b_dst = self.compute_nf_distance()		
			res = self.optimize_weights()
			sample_ens.append(res)
		res = pd.DataFrame(sample_ens)
		return res


	def load_raw_data(self):
		"""
			This method loads raw embedding and node-feature data.
			It will also extract the embedding and node-feature columns.
		"""
		self.emb = pd.read_csv(self.config["embedding_path"])
		self.emb.sort_values(by=["node"], inplace=True)
		self.nf = pd.read_csv(self.config["node_features_path"])
		self.nf.sort_values(by=["node"], inplace=True)
		self.graph_size = len(self.emb)
		# Get embedding columns:
		for col in self.emb.columns:
			if "emb_" in col:
				self.emb_cols.append(col)
		# Get node feature columns:
		for col in self.nf.columns:
			if "nf_" in col:
				self.nf_cols.append(col)
		if "node_features" in self.config:
			self.nf_cols = self.config["node_features"]
		if (len(self.emb_cols) == 0) or (len(self.nf_cols) == 0):
			raise ValueError("Failed to set emb and/or node feature columns.")


	def normalize_data(self):
		"""
			Normalizing data based on Ranking.
		"""
		node_label = self.emb["node"].values
		scaler = StandardScaler()
		norm_emb = scaler.fit_transform(self.emb[self.emb_cols].values)
		self.emb = pd.DataFrame(norm_emb)
		self.emb.columns = self.emb_cols
		self.emb["node"] = node_label

		node_label = self.nf["node"].values
		scaler = StandardScaler()
		norm_emb = scaler.fit_transform(self.nf[self.nf_cols].values)
		self.nf = pd.DataFrame(norm_emb)
		self.nf.columns = self.nf_cols
		self.nf["node"] = node_label


	def k_means_clustering(self):
		nf_kmeans = KMeans(
			n_clusters=self.config["k_means_numb_clusters"],
			random_state=42).fit_predict(self.nf[self.nf_cols].values)
		self.nf["cluster"] = nf_kmeans


	def get_node_feature_samples(self):
		p = self.config["sampling_fraction"]
		clusters = self.nf["cluster"].unique().tolist()
		# Get sampling size
		if self.config["sample_size"] == "default":
			sample_size = int(min([10**5, (len(self.nf)**2)/len(clusters)]))
		else:
			sample_size = int(self.config["sample_size"])
		# Get node cluster mapping
		node_cluster_map = {}
		node_cluster_frac = {}
		cluster_range = {}
		all_cluster_range = []
		for cluster in clusters:
			node_cluster_map[cluster] = self.nf[self.nf["cluster"] == cluster]["node"].tolist()
			node_cluster_frac[cluster] = float(len(node_cluster_map[cluster]))/self.graph_size
			cluster_range[cluster] = [cluster]*len(node_cluster_map[cluster])
			all_cluster_range += [cluster]*len(node_cluster_map[cluster])
	
		# Get sample nodes
		within_nodes = []
		between_nodes = []
		selected_pairs = {}
		for i in range(0, sample_size):
			rand = random.uniform(0, 1)
			if rand <= p:
				try:
					# Sample within cluster
					cluster = random.choice(all_cluster_range)
					nodes = random.sample(node_cluster_map[cluster], 2)
					if (nodes[0] != nodes[1]) and (str(nodes[1])+str(nodes[0]) not in selected_pairs):
						within_nodes.append((nodes[0], nodes[1]))
						selected_pairs[str(nodes[0])+str(nodes[1])] = None
						selected_pairs[str(nodes[1])+str(nodes[0])] = None
				except:
					pass
			else:
				try:
					# Sample between cluster
					cluster1 = random.choice(all_cluster_range)
					remaining_clusters = []
					for ii in cluster_range:
						if ii != cluster1:
							remaining_clusters += cluster_range[ii]
					cluster2 = random.choice(remaining_clusters)
					node1 = random.sample(node_cluster_map[cluster1], 1)[0]
					node2 = random.sample(node_cluster_map[cluster2], 1)[0]
					if (node1 != node2) and (str(node1)+str(node2) not in selected_pairs):
						between_nodes.append((node1, node2))
						selected_pairs[str(node1)+str(node2)] = None
						selected_pairs[str(node2)+str(node1)] = None
				except:
					pass

		return within_nodes, between_nodes


	def compute_nf_distance(self):
		# Get node feature mapping
		features = self.nf[self.nf_cols].values
		nodes = self.nf["node"].tolist()
		nf_mapping = {}
		for i in range(len(nodes)):
			nf_mapping[nodes[i]] = features[i]
		# Compute within distance
		within_distance = []
		for pairs in self.within_nodes:
			dist = np.power((nf_mapping[pairs[0]] - nf_mapping[pairs[1]]), 2)
			dist = math.sqrt(dist.sum())
			within_distance.append(dist)
		# Compute between distance
		between_distance = []
		for pairs in self.between_nodes:
			dist = np.power((nf_mapping[pairs[0]] - nf_mapping[pairs[1]]), 2)
			dist = math.sqrt(dist.sum())
			between_distance.append(dist)
		return within_distance, between_distance


	def compute_emb_distance(self, w=None):
		# Check weights
		if w is None:
			w = np.ones(len(self.emb_cols))
		# Get node embedding mapping
		embeddings = self.emb[self.emb_cols].values
		nodes = self.emb["node"].tolist()
		emb_mapping = {}
		for i in range(len(nodes)):
			emb_mapping[nodes[i]] = embeddings[i]
		# Compute within distance
		within_distance = []
		for pairs in self.within_nodes:
			dist = w * np.power((emb_mapping[pairs[0]] - emb_mapping[pairs[1]]), 2)
			dist = math.sqrt(dist.sum())
			within_distance.append(dist)
		# Compute between distance
		between_distance = []
		for pairs in self.between_nodes:
			dist = w * np.power((emb_mapping[pairs[0]] - emb_mapping[pairs[1]]), 2)
			dist = math.sqrt(dist.sum())
			between_distance.append(dist)
		return within_distance, between_distance


	def cost_func_distance(self, w=None):
		self.emb_w_dst, self.emb_b_dst = self.compute_emb_distance(w)
		x = np.array(self.nf_w_dst + self.nf_b_dst)
		y = np.array(self.emb_w_dst + self.emb_b_dst)
		z, p_val = pearsonr(x, y)
		z = 1 - pow(z, 2)
		return z


	def optimize_weights(self):
		pre_list = []
		post_list = []
		weight_list = []
		for i in range(1):
			# Initialize weights
			w_initial = np.random.rand(len(self.emb_cols))
			w_initial = np.divide(w_initial, w_initial.sum())
			# Set bounds for the weights
			w_bounds = np.array([[0, 1]] * len(self.emb_cols))
			# Run optimization
			score_pre_optimization = self.cost_func_distance()
			optm_res = minimize(self.cost_func_distance, w_initial, method='L-BFGS-B', bounds=w_bounds, tol=1e-5)
			score_post_optimization = self.cost_func_distance(optm_res.x)
			pre_list.append(score_pre_optimization)
			post_list.append(score_post_optimization)
			optm_res.x = np.divide(optm_res.x, optm_res.x.sum())
			weight_list.append(optm_res.x)

		pre_list = np.array(pre_list)
		post_list = np.array(post_list)

		weight_list_mean = np.mean(np.array(weight_list), axis=0)
		score_pre_optimization_mean = pre_list.mean()
		score_post_optimization_mean = post_list.mean()
		score_pre_optimization_std = np.std(pre_list)
		score_post_optimization_std = np.std(post_list)

		res = {}
		res["score_pre_optimization_mean"] = score_pre_optimization_mean
		res["score_post_optimization_mean"] = score_post_optimization_mean
		res["score_pre_optimization_std"] = score_pre_optimization_std
		res["score_post_optimization_std"] = score_post_optimization_std
		for i in range(len(self.emb_cols)):
			col = self.emb_cols[i]
			weight = weight_list_mean[i]
			res[col] = weight
		return res