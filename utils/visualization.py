from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from agents.user_agent import BotAgent, InfluencerAgent
import solara
import matplotlib.pyplot as plt
import networkx as nx
import community  # python-louvain package
from utils.metrics import calculate_agent_echo_chamber, calculate_content_propagation_clustering
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
    plt.figure(figsize=(12, 8))
    
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
    plt.title(f'Network Communities (Total: {num_communities})\nClustering Coefficient: {clustering_coef:.3f}\n'
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

@solara.component
def EchoChamberDashboard(model):
    """Dashboard component showing echo chamber metrics and visualizations"""
    # This is required to trigger updates when the model changes
    from mesa.visualization.utils import update_counter
    update_counter.get()
    
    # Get the latest data from the model
    if hasattr(model, 'datacollector') and model.datacollector.model_vars:
        # Check if data has been collected
        if "Echo_Chamber_Effect" in model.datacollector.model_vars:
            # Get the latest values
            latest_step = len(model.datacollector.model_vars["Echo_Chamber_Effect"]) - 1
            
            if latest_step >= 0:  # Make sure we have at least one data point
                echo_effect = model.datacollector.model_vars["Echo_Chamber_Effect"][latest_step]
                
                # Calculate the component metrics
                preference_scores = [calculate_agent_echo_chamber(a) 
                                    for a in model.agents if hasattr(a, "recommended_content") and len(a.recommended_content) > 0]
                avg_preference_score = sum(preference_scores) / len(preference_scores) if preference_scores else 0
                
                propagation_score = calculate_content_propagation_clustering(model)
                
                with solara.Column():
                    solara.Markdown("## Echo Chamber Analysis")
                    
                    with solara.Row():
                        with solara.Card(title="Overall Echo Chamber Effect"):
                            level = "Strong" if echo_effect > 0.7 else "Moderate" if echo_effect > 0.4 else "Weak"
                            solara.Markdown(f"**{echo_effect:.2f}** ({level})")
                    
                    with solara.Row():
                        with solara.Card(title="Preference Similarity"):
                            pref_level = "High" if avg_preference_score > 0.7 else "Medium" if avg_preference_score > 0.4 else "Low"
                            solara.Markdown(f"**{avg_preference_score:.2f}** ({pref_level})")
                            solara.Markdown("*How similar recommended content is to user preferences*")
                        
                        with solara.Card(title="Content Propagation Clustering"):
                            prop_level = "High" if propagation_score > 0.7 else "Medium" if propagation_score > 0.4 else "Low"
                            solara.Markdown(f"**{propagation_score:.2f}** ({prop_level})")
                            solara.Markdown("*How much content stays within network communities*")
                    
                    # Add a visualization of the echo chamber distribution
                    # if preference_scores:
                    # Use FigureMatplotlib instead of FigureContainer
                    # solara.FigureMatplotlib(create_echo_chamber_histogram(preference_scores))
            else:
                solara.Markdown("No data points collected yet. Run the simulation to see metrics.")
        else:
            solara.Markdown("Echo chamber metrics not available. Run the simulation to see metrics.")
    else:
        solara.Markdown("No data available yet. Run the simulation to see metrics.")


@solara.component
def EchoChamberNetwork(model):
    """Visualize the network with echo chamber highlighting"""
    visualize_echo_chamber_network(model)


def visualize_echo_chamber_network(model):
    """
    Visualize the network with communities and echo chamber effects highlighted
    """
    # Get the directed network
    directed_network = model.social_media_platform.social_network.network
    agent_types = get_agent_types(model)
    
    plt.figure(figsize=(8, 4))
    
    # Convert to undirected graph ONLY for community detection
    undirected_network = directed_network.to_undirected()
    
    # Detect communities using Louvain method on undirected network
    communities = community.best_partition(undirected_network)
    
    # Count the number of communities
    num_communities = len(set(communities.values()))
    
    # Get position layout that groups communities together
    pos = nx.spring_layout(directed_network, seed=42)  # Fixed seed for consistent layout
    
    # Calculate echo chamber scores for each agent
    echo_scores = {}
    for agent in model.agents:
        if hasattr(agent, "recommended_content") and agent.recommended_content:
            echo_scores[agent.pos] = calculate_agent_echo_chamber(agent)
        else:
            echo_scores[agent.pos] = 0
    
    # Define colors for agent types
    type_colors = {
        'influencer': '#d057d9',  
        'bot': '#53b028',        
        'user': '#4e6ac7'    
    }
    
    # Draw nodes with size based on echo chamber score
    node_colors = []
    node_sizes = []
    
    for node in directed_network.nodes():
        # Base color on agent type
        node_colors.append(type_colors[agent_types[node]])
        
        # Size based on echo chamber score (larger = stronger echo chamber)
        base_size = 100
        echo_factor = echo_scores.get(node, 0) * 2  # Scale up for visibility
        node_sizes.append(base_size * (1 + echo_factor))
    
    # Draw nodes
    nx.draw_networkx_nodes(directed_network, pos, 
                         node_color=node_colors, 
                         node_size=node_sizes,
                         alpha=0.8)
    
    # Draw edges with color based on whether they connect same community
    edge_colors = []
    edge_widths = []
    
    for edge in directed_network.edges():
        source, target = edge
        # Check if nodes are in the same community
        if communities[source] == communities[target]:
            # Same community - stronger echo chamber connection
            edge_colors.append('#ff9999')  # Light red
            edge_widths.append(1.5)
        else:
            # Different communities - weaker echo chamber connection
            edge_colors.append('#cccccc')  # Light gray
            edge_widths.append(0.5)
    
    # Draw edges
    nx.draw_networkx_edges(directed_network, pos, 
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.6,
                          arrows=True,
                          arrowsize=10)
    
    # Calculate average followers for each agent type DIRECTLY from the network
    # This is the critical part that's likely causing the issue
    num_influencers = int(model.influencer_percentage * model.num_agents)
    num_bots = int(model.bot_percentage * model.num_agents)
    num_agents = model.num_agents
    
    # Group nodes by type
    bot_indices = list(range(num_agents - num_bots, num_agents))
    user_indices = list(range(num_influencers, num_agents - num_bots))
    influencer_indices = list(range(num_influencers))
    
    # Calculate follower counts using in_degree on the DIRECTED network
    bot_followers = [directed_network.in_degree(i) for i in bot_indices]
    user_followers = [directed_network.in_degree(i) for i in user_indices]
    influencer_followers = [directed_network.in_degree(i) for i in influencer_indices]
    
    avg_bot_followers = sum(bot_followers) / len(bot_followers) if bot_followers else 0
    avg_user_followers = sum(user_followers) / len(user_followers) if user_followers else 0
    avg_influencer_followers = sum(influencer_followers) / len(influencer_followers) if influencer_followers else 0
    
    # Add legend with average followers information
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors['influencer'], 
                  markersize=10, label=f'Influencer (avg followers: {avg_influencer_followers:.1f})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors['bot'], 
                  markersize=10, label=f'Bot (avg followers: {avg_bot_followers:.1f})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors['user'], 
                  markersize=10, label=f'Regular User (avg followers: {avg_user_followers:.1f})'),
        plt.Line2D([0], [0], color='#ff9999', lw=2, label='Same Community'),
        plt.Line2D([0], [0], color='#cccccc', lw=1, label='Cross Community')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=6)
    
    # Add title with metrics
    plt.title(f'Echo Chamber Network Visualization\n'
             f'Number of communities: {num_communities}\n')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

@solara.component
def MetricsTrendDashboard(model):
    """Dashboard component showing trends of fake news and echo chamber metrics over time"""
    # This is required to trigger updates when the model changes
    from mesa.visualization.utils import update_counter
    update_counter.get()
    
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


def create_metrics_line_chart(model, metric_names, title):
    """Create a line chart for the given metrics"""
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
    
    # Create a custom layout component that organizes the dashboard in a grid
    @solara.component
    def DashboardLayout(model):
        # Apply some basic styling with inline styles
        container_style = {"max-width": "100%", "margin": "0 auto", "padding": "20px"}
        row_style = {"display": "flex", "margin-bottom": "20px", "width": "100%"}
        column_style = {"min-width": "100%"}
        
        with solara.Column(style=container_style):
            with solara.Row(style=row_style):
                with solara.Column(style=column_style):
                    with solara.Card(title="Project Information", style={"width": "fit-content", "height": "100%"}):
                        ProjectInfo(model)
                with solara.Column(style=column_style):
                    with solara.Card(style={"width": "100%", "height": "100%"}):
                        EchoChamberNetwork(model)
            with solara.Row(style=row_style):
                with solara.Column(style=column_style):
                    with solara.Card():
                        MetricsTrendDashboard(model)
                
                with solara.Column(style=column_style):
                    with solara.Card(title="Agent States"):
                        make_plot_component(["Number_of_Infected", "Number_of_Susceptible", "Number_of_Exposed"])(model)
            
    
    # Use the custom layout component
    viz = SolaraViz(
        model=model,
        components=[DashboardLayout],
        model_params=model_params
    )
    return viz


