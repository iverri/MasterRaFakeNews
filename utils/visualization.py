from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from agents.user_agent import BotAgent, InfluencerAgent
import solara
import matplotlib.pyplot as plt
import networkx as nx
import community  # python-louvain package
from utils.model_utils import get_agent_types
from recommender.types import RecommenderType


project_info = """
    # Recommender systems and fake news
    
    This is a simulation of how recommender systems affect the spread of fake news in social networks.
    The project is a part of the master thesis by Anna Holden Jacobsen and Lise Jakobsen.
    
    ## Agent Types:
    - **Regular users** (blue): Standard network participants
    - **Bot agents** (red): Automated accounts that spread content
    - **Influencer agents** (green): Users with high influence levels
    
    ## States:
    Agents transition through SIR-like states: Susceptible → Exposed → Infected
    
    *Created by Anna Holden Jacobsen and Lise Jakobsen - NTNU Trondheim*
    
    """

@solara.component
def ProjectInfo(model):
    project_info = model.info
    solara.Markdown(project_info)


@solara.component
def SocialNetwork(model):
    visualize_network(model.social_media_platform.social_network.network, get_agent_types(model))
   

def visualize_network(network, agent_types=None):
    """
    Visualize the network with communities and agent types
    
    Parameters:
    - network: networkx graph object
    - agent_types: dictionary mapping node ids to agent types ('influencer', 'bot', 'user')
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to undirected graph for community detection
    undirected_network = network.to_undirected()
    
    # Detect communities using Louvain method on undirected graph
    communities = community.best_partition(undirected_network)
    
    # Get position layout that groups communities together
    pos = nx.spring_layout(network)
    
    # Define colors for agent types - using lighter colors
    type_colors = {
        'influencer': '#d057d9',  
        'bot': '#53b028',        
        'user': '#4e6ac7'    
    }
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    
    num_agents = network.number_of_nodes()
    
    for node in network.nodes():
        # Set node size based on agent type and in-degree (number of followers)
        if agent_types:
            base_size = 100
            in_degree = network.in_degree(node)
            in_degree_factor = min(in_degree / (num_agents * 0.1), 2.0)  # Cap the scaling factor
            
            if agent_types[node] == 'influencer':
                node_colors.append(type_colors['influencer'])
                node_sizes.append(base_size * 1.5 * (1 + in_degree_factor * 0.5))  # Reduced size multiplier
            elif agent_types[node] == 'bot':
                node_colors.append(type_colors['bot'])
                node_sizes.append(base_size * 0.7 * (1 + in_degree_factor * 0.3))
            else:
                node_colors.append(type_colors['user'])
                node_sizes.append(base_size * (1 + in_degree_factor * 0.3))
        else:
            # If no agent types provided, color by community
            node_colors.append(communities[node])
            node_sizes.append(100)
    
    # Draw the network
    nx.draw_networkx_nodes(network, pos, 
                         node_color=node_colors, 
                         node_size=node_sizes)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(network, pos, 
                          alpha=0.2,
                          arrows=True,  # Show direction of edges
                          arrowsize=10)  # Size of arrow head
    
    # Add labels for agent types if provided
    if agent_types:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=type_colors[type_name], markersize=15, 
                      label=f'{type_name} (avg followers: {_get_avg_followers(network, type_name, agent_types):.1f})')
            for type_name in ['influencer', 'bot', 'user']
        ]
        plt.legend(handles=legend_elements, loc='upper left')
    
    # Add title with metrics
    clustering_coef = nx.average_clustering(undirected_network)
    modularity = community.modularity(communities, undirected_network)
    plt.title(f'Network Communities\nClustering Coefficient: {clustering_coef:.3f}\n'
             f'Modularity: {modularity:.3f}\n'
             f'Total Connections: {network.number_of_edges()}')
    
    plt.axis('off')
    plt.show()


def _get_avg_followers(network, agent_type, agent_types):
    """Helper method to calculate average followers for each agent type"""
    followers = [network.in_degree(node) 
                for node, type_ in agent_types.items() 
                if type_ == agent_type]
    return sum(followers) / len(followers) if followers else 0


def agent_portrayal(agent):
    """Define how to portray each agent"""
    portrayal = {
        "color": "blue",
        "size": 5,
    }

    if isinstance(agent, BotAgent):
        portrayal["color"] = "red"
    elif isinstance(agent, InfluencerAgent):
        portrayal["color"] = "green"
    
    portrayal["size"] *= agent.influence_level * 10
    return portrayal

# Model parameters for the UI controls
model_params = {
    "N": {
        "type": "SliderInt",
        "value": 100,
        "label": "Number of agents",
        "min": 50,
        "max": 500,
        "step": 1
    },
    "m_links": {
        "type": "SliderInt",
        "value": 10,
        "label": "Number of edges per new node",
        "min": 5,
        "max": 20,
        "step": 1
    },
    "influencer_percentage": {
        "type": "SliderInt",
        "value": 5,
        "label": "Percentage of influencers",
        "min": 1,
        "max": 10,
        "step": 1
    },
    "bot_percentage": {
        "type": "SliderInt",
        "value": 5,
        "label": "Percentage of bots",
        "min": 1,
        "max": 10,
        "step": 1
    },
    "news_amount": {
        "type": "SliderInt",
        "value": 500,
        "label": "Number of news items",
        "min": 100,
        "max": 1000,
        "step": 1,
        "default": 200
    },
    "fake_news_percentage": {
        "type": "SliderInt",
        "value": 10,
        "label": "Percentage of fake news",
        "min": 5,
        "max": 40,
        "step": 1
    },
    "recommender_type": {
        "type": "Select",
        "value": RecommenderType.RANDOM.value,
        "label": "Type of recommender",
        "values": [type.value for type in RecommenderType]
    },
    "num_recommendations": {
        "type": "SliderInt",
        "value": 10,
        "label": "Number of recommendations",
        "min": 5,
        "max": 20,
        "step": 1
    },
    "diversity_lambda": {
        "type": "SliderInt",
        "value": 0.1,
        "label": "Diversity lambda",
        "min": 0,
        "max": 1,
        "step": 0.01
    },
    "increase_diversity": {
        "type": "Checkbox",
        "value": False,
        "label": "Increase diversity"
    },
    "use_stored_network": {
        "type": "Checkbox",
        "value": False,
        "label": "Use stored network"
    }
}

def create_visualization(model_class):
    """Create a visualization for the given model class"""
    model = model_class()
    
    viz = SolaraViz(
        model=model,
        components=[
            ProjectInfo,
            make_plot_component(["Number_of_Infected", "Number_of_Susceptible", "Number_of_Exposed"]),
            SocialNetwork,
            make_plot_component(["Average_Diversity_Score"]),
            #make_plot_component(["Average_Feed_Size"]),
            
        ],
        model_params=model_params
    )
    return viz
