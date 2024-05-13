<h1 align="center">TravelPlanner<br> A Benchmark for Real-World Planning<br> with Language Agents </h1>

![Travel Planner](https://img.shields.io/badge/Task-Planning-blue)
![Travel Planner](https://img.shields.io/badge/Task-Tool_Use-blue) 
![Travel Planner](https://img.shields.io/badge/Task-Language_Agents-blue)  
![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![LLMs](https://img.shields.io/badge/Model-LLMs-green)

<p align="center">
    <img src="images/icon.png" width="10%"> <br>
</p>

Code for the Paper "[TravelPlanner: A Benchmark for Real-World Planning with Language Agents](http://arxiv.org/abs/2402.01622)".

![Demo Video GIF](images/TravelPlanner.gif)

<p align="center">
[<a href="https://osu-nlp-group.github.io/TravelPlanner/">Website</a>]•
[<a href="http://arxiv.org/abs/2402.01622">Paper</a>] •
[<a href="https://huggingface.co/datasets/osunlp/TravelPlanner">Dataset</a>] •
[<a href="https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard">Leaderboard</a>] •
[<a href="https://huggingface.co/spaces/osunlp/TravelPlannerEnvironment">Environment</a>] •
[<a href="https://twitter.com/ysu_nlp/status/1754365367294562680">Twitter</a>]
</p>

## Updates

- 2024/4/28: Update the [warnings](https://github.com/OSU-NLP-Group/TravelPlanner/tree/main?tab=readme-ov-file#%EF%B8%8Fwarnings), please note that we strictly prohibit any form of cheating.
- 2024/4/21: Provide [format check tool](./postprocess/format_check.py)  for testset submission files.  You can run it to check if there are any format errors in your file.

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
# if you do not want to test google models, like gemini, just input "1".
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
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# SET_TYPE in ['validation', 'test']
export SET_TYPE=validation
# STRATEGY in ['direct','cot','react','reflexion']
export STRATEGY=direct

cd tools/planner
python sole_planning.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY
```

## Postprocess

In order to parse natural language plans, we use gpt-4 to convert these plans into json formats. We encourage developers to try different parsing prompts to obtain better-formatted plans.

```bash
export OUTPUT_DIR=path/to/your/output/file
export MODEL_NAME=MODEL_NAME
export OPENAI_API_KEY=YOUR_OPENAI_KEY
export SET_TYPE=validation
export STRATEGY=direct
# MODE in ['two-stage','sole-planning']
export MODE=two-stage
export TMP_DIR=path/to/tmp/parsed/plan/file
export SUBMISSION_DIR=path/to/your/evaluation/file

cd postprocess
python parsing.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --mode $MODE --tmp_dir $TMP_DIR

# Then these parsed plans should be stored as the real json formats.
python element_extraction.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --mode $MODE --tmp_dir $TMP_DIR

# Finally, combine these plan files for evaluation. We also provide a evaluation example file "example_evaluation.jsonl" in the postprocess folder.
python combination.py --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --mode $MODE  --submission_file_dir $SUBMISSION_DIR
```

## Evaluation

We support the offline validation set evaluation through the provided evaluation script. To avoid data contamination, please use our official [leaderboard](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard) for test set evaluation.

```bash
export SET_TYPE=validation
export EVALUATION_FILE_PATH=your/evaluation/file/path

cd evaluation
python eval.py --set_type $SET_TYPE --evaluation_file_path $EVALUATION_FILE_PATH
```

## ⚠️Warnings

We release our evaluation scripts to foster innovation and aid the development of new methods.  We encourage the use of evaluation feedback in training set, such as implementing reinforcement learning techniques, to enhance learning. However, we strictly prohibit any form of cheating in the validation and test sets to uphold the fairness and reliability of the benchmark's evaluation process. We reserve the right to disqualify results if we find any of the following violations:

1. Reverse engineering of our dataset, which includes, but is not limited to:
   - Converting our natural language queries in the test set to structured formats (e.g., JSON) for optimization and unauthorized evaluation.
   - Deriving data point entries using the hard rules from our data construction process, without accessing the actual database.
   - Other similar manipulations.
2. Hard coding or explicitly writing evaluation cues into prompts by hand, such as direct hints of common sense, which contradicts our goals as it lacks generalizability and is limited to this specific benchmark.
3. Any other human interference strategies that are tailored specifically to this benchmark but lack generalization capabilities.

(The content above is intended solely for use within the TravelPlanner evaluation framework. Extending and editing our database to create new tasks or benchmarks is permitted, provided that you adhere to the licensing terms.)

## Load Datasets

```python
from datasets import load_dataset
# test can be substituted by "train" and "validation".
data = load_dataset('osunlp/TravelPlanner','test')['test']
```

## TODO

- ##### Code

  - [x] Baseline Code

  - [x] Query Construction Code

  - [x] Evaluation Code
  - [x] Plan Parsing and Element Extraction Code

- ##### Environment

  - [x] Release Environment Database
  - [ ] Database Field Introduction

## Contact

If you have any problems, please contact 
[Jian Xie](mailto:jianx0321@gmail.com),
[Kai Zhang](mailto:zhang.13253@osu.edu),
[Yu Su](mailto:su.809@osu.edu)

## Citation Information

If our paper or related resources prove valuable to your research, we kindly ask for citation. 

<a href="https://github.com/OSU-NLP-Group/TravelPlanner"><img src="https://img.shields.io/github/stars/OSU-NLP-Group/TravelPlanner?style=social&label=TravelPanner" alt="GitHub Stars"></a>

```
@article{xie2024travelplanner,
  title={Travelplanner: A benchmark for real-world planning with language agents},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Zhu, Tinghui and Lou, Renze and Tian, Yuandong and Xiao, Yanghua and Su, Yu},
  journal={arXiv preprint arXiv:2402.01622},
  year={2024}
}
```
