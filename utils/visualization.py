from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from agents.user_agent import BotAgent, InfluencerAgent

def agent_portrayal(agent):
    """Define how to portray each agent"""
    if isinstance(agent, BotAgent):
        color = "red"
    elif isinstance(agent, InfluencerAgent):
        color = "green"
    else:
        color = "blue"
    
    return {
        "color": color,
        "size": 50 * agent.influence_level
    }

# Model parameters for the UI controls
model_params = {
    "N": {
        "type": "SliderInt",
        "value": 5,
        "label": "Number of agents",
        "min": 5,
        "max": 100,
        "step": 1
    },
    "m_links": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of edges per new node",
        "min": 1,
        "max": 5,
        "step": 1
    }
}

def create_visualization(model_class):
    """Create a visualization for the given model class"""
    # Create initial model instance
    model = model_class(N=5, m_links=1)
    
    # Create visualization with explicit components
    viz = SolaraViz(
        model=model,
        components=[
            make_space_component(agent_portrayal),
            make_plot_component(["Number_of_Believers", "Number_of_Susceptible", "Number_of_Exposed"]),
            make_plot_component(["Average_Clustering", "Community_Modularity"])
        ],
        model_params=model_params
    )
    return viz 