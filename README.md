# MasterRaFakeNews

Master thesis repository for Anna Holden Jacobsen and Lise Jakobsen - Computer Science students at NTNU Trondheim.

## Project Description

An agent-based model (ABM) simulation of fake news propagation in social networks. The model simulates how different types of agents (regular users, bots, and influencers) interact with and spread news content through a social network structure.

Diagrams can be found in the [diagrams](diagrams) folder.

## Features

- Agent-based modeling using Mesa framework
- Social network simulation using NetworkX
- Different agent types:
  - Regular users
  - Bot agents
  - Influencer agents
- News content classification (real/fake)
- Network metrics and visualization
- SIR-like state transitions (Susceptible → Exposed → Infected)
- Preference-based network formation
- Community detection using Louvain method

## Requirements

- pandas
- numpy
- networkx
- mesa
- lenskit
- altair
- matplotlib
- seaborn
- solara
- python-louvain

## Installation

1. Clone the repository:

```bash
git clone https://github.com/annajacobsen/MasterRaFakeNews.git
cd MasterRaFakeNews
```

2. Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the simulation:

```bash
solara run model.py
```

This will launch a Solara visualization interface where you can:
- Adjust the number of agents (5-100)
- Modify the number of edges per new node (1-5)
- View network visualization
- Monitor metrics like:
  - Number of Infected/Susceptible/Exposed agents
  - Average clustering
  - Community modularity

## Project Structure

- `model.py`: Main simulation model
- `agents/`: Agent definitions and behaviors
  - `user_agent.py`: Regular, Bot, and Influencer agent implementations
- `objects/`: Core simulation objects
  - `news_content.py`: News content representation
  - `social_network.py`: Network structure and dynamics
  - `social_media_platform.py`: Platform that connects network and recommender
- `recommender/`: Content recommendation system
- `utils/`: Helper functions and utilities
  - `metrics.py`: Network analysis metrics
  - `visualization.py`: Visualization components
  - `similarity.py`: Content similarity calculations
  - `agents_utils.py`: Agent behavior utilities
  - `objects_utils.py`: Network creation and manipulation
  - `model_utils.py`: Model initialization helpers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Anna Holden Jacobsen
- Lise Jakobsen

Department of Computer Science  
Norwegian University of Science and Technology (NTNU)  
Trondheim, Norway
