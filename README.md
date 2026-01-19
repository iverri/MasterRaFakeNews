# Fake News Propagation in Social Networks: An Agent-Based Model

Master thesis repository for Anna Johanne Holden Jacobsen and Lise Jakobsen - Computer Science students at NTNU Trondheim.

## Project Description

This project implements an agent-based model (ABM) to simulate the propagation of fake news in social networks. The simulation explores how different recommendation algorithms affect the spread of misinformation among users with varying characteristics.

### Key Features

- **Agent Types**: Regular users, bot agents, and influencer agents with distinct behaviors
- **SEIS-like State Model**: Agents transition through Susceptible → Exposed → Infected states
- **Multiple Recommendation Algorithms**: Random, collaborative filtering, content-based, and popularity-based
- **Social Network Dynamics**: Preference-based network formation with realistic follower distributions
- **Comprehensive Metrics**: Echo chamber effects, misinformation spread, misinformation exposure, diversity scores
- **Interactive Dashboard**: Real-time visualization and parameter adjustment
- **Batch Experiments**: Systematic comparison of recommendation algorithms

### Model Architecture

FakeNewsModel
```
├── SocialMediaPlatform
│ ├── SocialNetwork (NetworkX directed graph)
│ └── Recommender (Multiple algorithm implementations)
├── Agents (UserAgent, BotAgent, InfluencerAgent)
├── NewsContent (Real/fake news with topic vectors)
└── DataCollector (Metrics and visualization)
```



## Dashboard

![Solara Dashboard](diagrams/solara_dashboard.png)

The Solara dashboard provides an interactive interface to run and visualize the simulation. It allows you to:
- Adjust simulation parameters in real-time
- View the network graph with different agent types and states
- Monitor key metrics as the simulation progresses
- Observe how misinformation spreads through the network

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
- dppy
- scikit-learn


## Project Structure

```
├── model.py # Main simulation model
├── config.py # Configuration parameters
├── agents/
│ └── user_agent.py # Agent implementations
├── objects/
│ ├── news_content.py # News content and generation
│ ├── social_network.py # Network structure
│ └── social_media_platform.py # Platform orchestration
├── recommender/
│ ├── recommender.py # Recommendation algorithms
│ └── types.py # Algorithm type definitions
├── utils/
│ ├── metrics.py # Evaluation metrics
│ ├── datacollector.py # Data collection setup
│ ├── visualization.py # Solara dashboard components
│ ├── objects_utils.py # Network creation utilities
│ ├── model_utils.py # Model initialization helpers
│ └── .py # Various utilities
├── experiments.py # Batch experiment runner
├── plot.py # Result visualization
├── experiment_results/ # Output directory
├── plots/ # Generated visualizations
└── diagrams/ # Documentation diagrams
```

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

For windows:

```bash
python3.11 -m venv .venv
.venv\Scripts\activate
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
  - Misinformation Ratio Difference
  - Misinformation Count in Recommendations

### Run experiments

```bash
python3 experiments.py
```
You can modify the parameters in the `experiments.py` file.

Experiment results are saved in the `experiment_results` folder.

### Plotting results
After running experiments, you can generate plots to visualize the results:

```bash
python3 plot.py experiment_results/{experiment_name}.csv --output-dir plots
```
Remember to replace `{experiment_name}` with the name of the experiment file you want to plot, along with the correct timestamp.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Anna Johanne Holden Jacobsen
- Lise Jakobsen

Department of Computer Science  
Norwegian University of Science and Technology (NTNU)  
Trondheim, Norway
