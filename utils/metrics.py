import networkx as nx


def get_network_metrics(G):
    """Calculate and return key network metrics"""
    metrics = {
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "degree_distribution": dict(G.degree()),
        "centrality": {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G),
            "eigenvector": nx.eigenvector_centrality(G, max_iter=1000)
        }
    }
    return metrics

def get_community_modularity(G):
    """Calculate community modularity using Louvain method"""
    communities = nx.community.louvain_communities(G)
    return nx.community.modularity(G, communities)