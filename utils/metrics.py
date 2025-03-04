import networkx as nx
import community  # python-louvain package

'''
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

'''

def get_community_modularity(G):
    """Calculate community modularity using Louvain method"""
    communities = nx.community.louvain_communities(G)
    return nx.community.modularity(G, communities)

def get_clustering_metrics(G):
    """
    Return detailed clustering metrics for a network.
    
    Args:
        G (nx.Graph): The network graph
        
    Returns:
        dict: Dictionary of network metrics
    """
    # Convert to undirected for metrics that require it
    if isinstance(G, nx.DiGraph):
        undirected_network = G.to_undirected()
    else:
        undirected_network = G
    
    metrics = {
        'average_clustering': nx.average_clustering(undirected_network),
        'clustering_by_node': nx.clustering(undirected_network),
        'communities': community.best_partition(undirected_network),
        'modularity': community.modularity(
            community.best_partition(undirected_network), 
            undirected_network
        ),
        'transitivity': nx.transitivity(undirected_network)
    }
    
    # Add directed-specific metrics if applicable
    if isinstance(G, nx.DiGraph):
        metrics.update({
            'in_degree_centrality': nx.in_degree_centrality(G),
            'out_degree_centrality': nx.out_degree_centrality(G),
            'average_in_degree': sum(dict(G.in_degree()).values()) / G.number_of_nodes(),
            'average_out_degree': sum(dict(G.out_degree()).values()) / G.number_of_nodes(),
        })
    
    return metrics