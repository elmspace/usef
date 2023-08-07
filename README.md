# USEF
Unsupervised Structural Embedding Framework

### Installing

* You can install USEF by running:
```
pip install git+https://github.com/elmspace/usef.git
```
* It used Python 3.x (I am using 3.9)
* It auto installs the following packages:
  * pandas==2.0.3
  * networkx==3.1
  * numpy==1.25.2
  * scipy==1.11.1
  * scikit-learn==1.3.0 

### Code Sample
from usef.usef import USEF

```
graph_size = 500

config = {}
config["k_means_numb_clusters"] = int(0.05 * graph_size)
config["sampling_fraction"] = 0.5
config["sample_size"] = int(0.5 * graph_size)
config["sample_ensemble"] = 50
config["node_features_path"] = "./data/nf.csv"
config["embedding_path"] = "./data/emb.csv"
config["node_features"] = ["nf_degree_centrality"]

obj = USEF(config)
res = obj.run()
```
