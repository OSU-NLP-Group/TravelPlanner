<h1 align="center">TravelPlanner<br> Toward Real-World Planning with Language Agents </h1>

![Travel Planner](https://img.shields.io/badge/Task-Planning-blue)
![Travel Planner](https://img.shields.io/badge/Task-Tool_Use-blue) 
![Travel Planner](https://img.shields.io/badge/Task-Language_Agents-blue)  
![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![LLMs](https://img.shields.io/badge/Model-LLMs-green)

<p align="center">
    <img src="images/icon.png" width="10%"> <br>
  Logo for <b>TravelPlanner</b>.
</p>

Code for the Paper "[TravelPlanner: Toward Real-World Planning with Language Agents]()".

![Demo Video GIF](images/TravelPlanner.gif)

<p align="center">
[<a href="https://osu-nlp-group.github.io/TravelPlanner/">Website</a>]•
[<a href="">Paper</a>] •
[<a href="https://huggingface.co/datasets/osunlp/TravelPlanner">Dataset</a>] •
[<a href="https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard">Leaderboard</a>] •
[<a href="https://huggingface.co/spaces/osunlp/TravelPlannerEnvironment">Environment</a>] •
[<a href="https://twitter.com/ysu_nlp/status/1742398541660639637">Twitter</a>]
</p>



# TravelPlanner

TravelPlanner is a benchmark crafted for evaluating language agents in tool-use and complex planning within multiple constraints.

For a given query, language agents are expected to formulate a comprehensive plan that includes transportation, daily meals, attractions, and accommodation for each day.

For constraints, from the perspective of real world applications, TravelPlanner includes three types of them: Environment Constraint, Commonsense Constraint, and Hard Constraint. 


## Setup Environment

1. Create a conda environment and install dependency:
```bash
conda create -n travelplanner python=3.9
conda activate travelplanner
pip install -r requirements.txt
```

2. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory (i.e., `your/path/TravelPlanner`).

## Running
### Two-stage Mode

In the two-stage mode, language agents are tasked to with employing various search tools to gather information.
Based on the collected information, language agents are expected to deliver a plan that not only meet the user’s needs specified in the query but also adheres to commonsense constraints.

```bash
export OUTPUT_DIR=path/to/your/output/file
# We support MODEL in ['gpt-3.5-turbo-X','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral']
export MODEL_NAME=MODEL_NAME
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google model, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# SET_TYPE in ['validation', 'test']
export SET_TYPE=validation
cd agents
python tool_agents.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME
```
The generated plan will be stored in OUTPUT_DIR/SET_TYPE.

### Sole-Planning Mode

TravelPlanner also provides an easier mode solely focused on testing their planning ability.
The sole-planning mode ensures that no crucial information is missed, thereby enabling agents to focus on planning itself.

Please refer to paper for more details.

```bash
export OUTPUT_DIR=path/to/your/output/file
# We support MODEL in ['gpt-3.5-turbo-X','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral']
export MODEL_NAME=MODEL_NAME
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google model, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# SET_TYPE in ['validation', 'test']
export SET_TYPE=validation
# STRATEGY in ['direct','cot','react','reflexion']
export STRATEGY=direct

cd tools/planner
python sole_planning.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY
```

## TODO

- ##### Code

  - [x] Baseline Code

  - [x] Query Construction Code

  - [x] Evaluation Code
  - [ ] Plan Parsing and Element Extraction Code

- ##### Environment

  - [x] Release Environment Database
  - [ ] Database Field Introduction

## Contact

If you have any problems, please contact 
[JIan Xie](mailto:jianx0321@gmial.com),
[Kai Zhang](mailto:zhang.13253@osu.edu),
[Yu Su](mailto:su.809@osu.edu)

## Citation Information

If our paper or related resources prove valuable to your research, we kindly ask for citation. 

<a href="https://github.com/OSU-NLP-Group/TravelPlanner"><img src="https://img.shields.io/github/stars/OSU-NLP-Group/TravelPlanner?style=social&label=TravelPanner" alt="GitHub Stars"></a>

```
@article{Xie2024TravelPlanner,
  author    = {},
  title     = {TravelPlanner: Toward Real-World Planning with Language Agents},
  journal   = {},
  year      = {2024}
}
```
