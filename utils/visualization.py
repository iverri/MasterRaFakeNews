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
    - **Bot agents** (green): Automated accounts that spread content
    - **Influencer agents** (pink): Users with high influence levels
    
    ## States:
    Agents transition through SIR-like states: Susceptible → Exposed → Infected, but they can go back to susceptible after a certain amount of time.
    
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
    plt.figure(figsize=(9, 5))
    
    # Convert to undirected graph for community detection
    undirected_network = network.to_undirected()
    
    # Detect communities using Louvain method on undirected graph
    communities = community.best_partition(undirected_network)
    
    # Count the number of communities
    num_communities = len(set(communities.values()))
    
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
    plt.title(f'Network Communities (Total: {num_communities})')
    
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
'''
@solara.component
def MisinformationDashboard(model):
    """Dashboard component showing misinformation metrics"""
    # This is required to trigger updates when the model changes
    from mesa.visualization.utils import update_counter
    update_counter.get()
    
    # Get the latest data from the model
    if hasattr(model, 'datacollector') and model.datacollector.model_vars:
        # Check if data has been collected
        if "Misinformation_Spread_Percentage" in model.datacollector.model_vars:
            # Get the latest values for each metric
            # In Mesa, model_vars stores data as a list where the index corresponds to the step
            latest_step = len(model.datacollector.model_vars["Misinformation_Spread_Percentage"]) - 1
            
            if latest_step >= 0:  # Make sure we have at least one data point
                misinfo_count = model.datacollector.model_vars["Misinformation_Count_In_Recommendations"][latest_step]
                ratio_diff = model.datacollector.model_vars["Misinformation_Ratio_Difference"][latest_step]
                spread_pct = model.datacollector.model_vars["Misinformation_Spread_Percentage"][latest_step] * 100
                echo_effect = model.datacollector.model_vars["Echo_Chamber_Effect"][latest_step]
                
                with solara.Column():
                    solara.Markdown("## Misinformation Metrics")
                    
                    with solara.Row():
                        with solara.Card(title="Fake News in Recommendations"):
                            solara.Markdown(f"**{misinfo_count:.2f}** items on average")
                        
                        with solara.Card(title="Recommendation Bias"):
                            bias_text = "higher" if ratio_diff > 0 else "lower"
                            solara.Markdown(f"**{abs(ratio_diff)*100:.1f}%** {bias_text} than content pool")
                    
                    with solara.Row():
                        with solara.Card(title="Population Exposed"):
                            solara.Markdown(f"**{spread_pct:.1f}%** of agents exposed")
                        
                        with solara.Card(title="Echo Chamber Effect"):
                            level = "Strong" if echo_effect > 0.7 else "Moderate" if echo_effect > 0.4 else "Weak"
                            solara.Markdown(f"**{echo_effect:.2f}** ({level})")
            else:
                solara.Markdown("No data points collected yet. Run the simulation to see metrics.")
        else:
            solara.Markdown("Misinformation metrics not available. Run the simulation to see metrics.")
    else:
        solara.Markdown("No data available yet. Run the simulation to see metrics.")

def create_echo_chamber_histogram(scores):
    """Create a histogram of echo chamber scores"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create histogram
    n, bins, patches = ax.hist(scores, bins=10, alpha=0.7, color='#4e6ac7')
    
    # Add a vertical line for the average
    avg = sum(scores) / len(scores)
    ax.axvline(x=avg, color='red', linestyle='--', linewidth=2, label=f'Average: {avg:.2f}')
    
    # Add labels and title
    ax.set_xlabel('Echo Chamber Score')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Distribution of Echo Chamber Effects Across Agents')
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

'''

    

def create_plot_with_cleanup(plot_function, *args, **kwargs):
    """Wrapper to ensure plots are properly cleaned up before creation."""
    plt.close('all')  # Close all existing figures
    return plot_function(*args, **kwargs)

def create_metrics_trend_plot(model, metrics):
    """Create a plot showing trends for multiple metrics."""
    plt.close('all')  # Close all existing figures
    # Rest of your function...

'''
@solara.component
def MetricsTrendDashboard(model):
    """Dashboard component showing trends of fake news and echo chamber metrics over time"""
    # This is required to trigger updates when the model changes
    from mesa.visualization.utils import update_counter
    update_id = update_counter.get()
    component_key = f"metrics_trend_{id(model)}_{update_id}"
    
    with solara.Column():
        # Check if data has been collected
        if hasattr(model, 'datacollector') and model.datacollector.model_vars:
            # Get the metrics we want to track
            metrics = {
                "Misinformation Metrics": [
                    "Misinformation_Ratio_Difference",
                    "Misinformation_Count_In_Recommendations",
                ],
            }
            
            # Check if we have enough data points
            if all(metric in model.datacollector.model_vars for metric in 
                   metrics["Misinformation Metrics"]):
                
                # Get the number of steps
                steps = len(model.datacollector.model_vars[metrics["Misinformation Metrics"][0]])
                
                if steps > 1:  # Need at least 2 points for a line chart
                    with solara.Column():
                        solara.Markdown("## Metrics Development Over Time")
                        
                        # Create misinformation metrics chart
                        solara.FigureMatplotlib(create_metrics_line_chart(
                            model, 
                            metrics["Misinformation Metrics"], 
                            "Misinformation Metrics Over Time"
                        ))
                        
                else:
                    solara.Markdown("Not enough data points yet. Run the simulation longer to see trends.")
            else:
                solara.Markdown("Some metrics are not available. Run the simulation to see trends.")
        else:
            solara.Markdown("No data available yet. Run the simulation to see trends.")
'''

def create_metrics_line_chart(model, metric_names, title):
    """Create a line chart for the given metrics"""
    # Close all existing figures to prevent accumulation
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Get the data for each metric
    x = list(range(len(model.datacollector.model_vars[metric_names[0]])))
    
    # Define colors for different metrics
    colors = ['#d057d9', '#53b028', '#4e6ac7', '#ff9999', '#cccccc']
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        y = model.datacollector.model_vars[metric]
        
        # Format the metric name for display
        display_name = metric.replace('_', ' ')
        
        # Scale percentage metrics to show as percentages
        if "Percentage" in metric or "Ratio" in metric:
            y = [val * 100 if val <= 1 else val for val in y]
            display_name += " (%)"
        
        ax.plot(x, y, label=display_name, color=colors[i % len(colors)], linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    # Set y-axis limits appropriately based on the chart type
    if "Misinformation" in title:
        # For misinformation metrics, ensure we can see small values clearly
        y_max = max(max(model.datacollector.model_vars[metric_names[0]]) * 100 * 1.1, 20)
        ax.set_ylim(bottom=-10, top=y_max)
    elif "Echo Chamber" in title:
        # For echo chamber metrics, use a 0-1 scale unless values exceed it
        y_values = []
        for metric in metric_names:
            y_values.extend(model.datacollector.model_vars[metric])
        y_max = max(max(y_values) * 1.1, 1.0)
        ax.set_ylim(bottom=0, top=y_max)
    
    plt.tight_layout()
    return fig

def create_content_propagation_plot(model):
    """Create a line plot for Content Propagation Clustering over time."""
    # Close all existing figures to prevent accumulation
    plt.close('all')
    
    if not hasattr(model, "datacollector"):
        return None
    df = model.datacollector.get_model_vars_dataframe()
    if "Content_Propagation_Clustering" not in df:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Content_Propagation_Clustering"], marker='o', color='#ff9999')
    ax.set_title("Content Propagation Clustering Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Within-Community Share Ratio")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig



def create_cluster_content_similarity_plot(model):
    """Plot within- and between-cluster content similarity over time."""
    # Close all existing figures to prevent accumulation
    plt.close('all')
    
    if not hasattr(model, "datacollector"):
        return None
    df = model.datacollector.get_model_vars_dataframe()
    if "Within_Cluster_Content_Similarity" not in df or "Between_Cluster_Content_Similarity" not in df:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Within_Cluster_Content_Similarity"], label="Within-Cluster Similarity", color="blue")
    ax.plot(df["Between_Cluster_Content_Similarity"], label="Between-Cluster Similarity", color="orange")
    ax.set_title("Cluster Content Similarity Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

@solara.component
def ClusterContentSimilarityDashboard(model):
    fig = create_cluster_content_similarity_plot(model)
    if fig is not None:
        solara.FigureMatplotlib(fig)
    else:
        solara.Markdown("No data available yet. Run the simulation to see cluster content similarity.")

def create_echo_chamber_strength_plot(model):
    """Plot echo chamber strength (difference and ratio) over time."""
    # Close all existing figures to prevent accumulation
    plt.close('all')
    
    if not hasattr(model, "datacollector"):
        return None
    df = model.datacollector.get_model_vars_dataframe()
    if "Echo_Chamber_Strength_Diff" not in df or "Echo_Chamber_Strength_Ratio" not in df:
        return None
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["Echo_Chamber_Strength_Diff"], label="Difference (Within - Between)", color="purple")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Difference", color="purple")
    ax1.tick_params(axis='y', labelcolor="purple")
    ax1.set_title("Echo Chamber Strength Over Time")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["Echo_Chamber_Strength_Ratio"], label="Ratio (Within / Between)", color="green")
    ax2.set_ylabel("Ratio", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    return fig

@solara.component
def EchoChamberStrengthDashboard(model):
    fig = create_echo_chamber_strength_plot(model)
    if fig is not None:
        solara.FigureMatplotlib(fig)
    else:
        solara.Markdown("No data available yet. Run the simulation to see echo chamber strength.")

def create_average_diversity_plot(model):
    """Plot average diversity score over time."""
    if not hasattr(model, "datacollector"):
        return None
    df = model.datacollector.get_model_vars_dataframe()
    if "Average_Diversity_Score" not in df:
        return None
    
    # Create a new figure and clear any previous plots
    plt.close('all')  # Close all existing figures
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(df["Average_Diversity_Score"], marker='o', color='#9370db')  # Medium purple color
    ax.set_title("Average Recommendation Diversity Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Diversity Score (0-1)")
    ax.grid(True, alpha=0.3)
    # Set y-axis limits to better visualize changes
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

@solara.component
def AverageDiversityDashboard(model):
    """Dashboard component for Average Diversity metric."""
    
    fig = create_average_diversity_plot(model)
    if fig is not None:
        solara.FigureMatplotlib(fig)
    else:
        solara.Markdown("No diversity data available yet. Run the simulation to see average diversity scores.")

# Model parameters for the UI controls
model_params = {
    "N": {
        "type": "SliderInt",
        "value": 200,
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
    "diversity_level": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Diversity level",
        "min": 0,
        "max": 1.0,
        "step": 0.01
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
    
    # Create a custom layout component that organizes the dashboard in a grid
    @solara.component
    def DashboardLayout(model):
        # Apply some basic styling with inline styles
        container_style = {"max-width": "100%", "margin": "0 auto", "padding": "10px"}
        row_style = {"display": "flex", "margin-bottom": "20px", "width": "100%"}
        column_style = {"min-width": "100%"}
        
        with solara.Column(style=container_style):
            with solara.Row(style=row_style):
                with solara.Column(style=column_style):
                    with solara.Card(title="Project Information", style={"width": "fit-content", "height": "100%"}):
                        ProjectInfo(model)
                with solara.Column(style=column_style):
                    with solara.Card(title="Social Network", style={"width": "fit-content", "height": "100%", "max-height": "500px"}):
                        SocialNetwork(model)
            with solara.Row(style=row_style):
                with solara.Column(style=column_style):
                    with solara.Card(title="Agent States"):
                        make_plot_component(["Number_of_Infected", "Number_of_Susceptible", "Number_of_Exposed"])(model)
                with solara.Column(style=column_style):
                    with solara.Card(title="Misinformation Metrics"):
                        make_plot_component(["Misinformation_Ratio_Difference", "Misinformation_Count_In_Recommendations"])(model)

    
    viz = SolaraViz(
        model=model,
        components=[DashboardLayout],
        model_params=model_params
    )
    return viz


