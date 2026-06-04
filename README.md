# Fake News Propagation in Social Networks: An Agent-Based Model

Master thesis repository for Iver Ringheim and Elena Willmann - Computer Science students at NTNU Trondheim. \
A fork from Anna Johanne Holden Jacobsen and Lise Jakobsen's [repository](https://github.com/lisejakobsen/MasterRaFakeNews).

## Project Description

This project expands upon the foundations created by Anna Johanne Holden Jacobsen and Lise Jakobsen.

### New additions
- **Personality**: Gives agents personalities that affect the social network structure and agent interactions with content.
- **Evaluation metrics**: Allows for evaluation of the quality of the recommendations provided by the recommendation algorithm.
- **Accuracy enhancements**: Fixes parts of the simulation that caused content-based and collaborative filtering to have lower accuracy than random.
- **Hybrid recommendation algorithms**: Expands the number of available recommendation algorithms by including algorithms used more in practice.
-  **Runtime enhancements**: Optimises the simulation to run faster and to be less resource-heavy.

### Key Features

- **Agent Types**: Regular users, bot agents, and influencer agents with distinct behaviors
- **SEIS-like State Model**: Agents transition through Susceptible → Exposed → Infected states
- **Multiple Recommendation Algorithms**: Random, collaborative filtering, content-based, popularity-based and hybrid approaches.
- **Social Network Dynamics**: Preference-based network formation with realistic follower distributions
- **Comprehensive Metrics**: misinformation spread, infection rate, misinformation count,
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


## Run experiments

```bash
python3 experiments.py
```
You can modify the parameters in the `experiments.py` file. \
If the number of agents exceed 200, it is suggested to switch off community data (collect_community_data = False)

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

- Iver Ringheim
- Elena Willmann
- Anna Johanne Holden Jacobsen
- Lise Jakobsen

Department of Computer Science  
Norwegian University of Science and Technology (NTNU)  
Trondheim, Norway
