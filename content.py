TITLE = """<h1 align="center" id="space-title">TravelBench Leaderboard</h1>"""

INTRODUCTION_TEXT = """
TravelBench is a benchmark crafted for evaluating language agents in tool-use and complex planning within multiple constraints. (See our [paper](https://arxiv.org/abs/2311.12983) for more details.)

## Data
In TravelBench, for a given query, language agents are expected to formulate a comprehensive plan that includes transportation, daily meals, attractions, and accommodation for each day.
For constraints, from the perspective of real world applications, we design three types of them: Environment Constraint, Commonsense Constraint, and Hard Constraint.
TravelBench comprises 1,225 queries in total. The number of days and hard constraints are designed to test agents' abilities across both the breadth and depth of complex planning.

TravelBench data can be found in [this dataset](https://huggingface.co/datasets/osunlp/TravelBench). 

## Submission Guidelines for TravelBench
Participants are invited to submit results for both validation and testing phases. The submissions will be evaluated based on several metrics: delivery rate, commonsense constraint pass rate (micro/macro), hard constraint pass rate (micro/macro), and the final pass rate.

### Format of Submission:
Submissions must be in the form of a JSON-line file. Each line should adhere to the following structure:
```
{"idx":0,"query":"Natural Language Query","plan":[{"day": 1, "current_city": "from [City A] to [City B]", "transportation": "Flight Number: XXX, from A to B", "breakfast": "Name, City", "attraction": "Name, City;Name, City;...;Name, City;", "lunch": "Name, City", "dinner": "Name, City", "accommodation": "Name, City"}, {"day": 2, "current_city": "City B", "transportation": "-", "breakfast": "Name, City", "attraction": "Name, City;Name, City;", "lunch": "Name, City", "dinner": "Name, City", "accommodation": "Name, City"}, ...]}
```
Explanation of Fields:
#### day:
Description: Indicates the specific day in the itinerary.
Format: Enter the numerical value representing the sequence of the day within the travel plan. For instance, '1' for the first day, '2' for the second day, and so on.

#### current city:
Description: Indicates the city where the traveler is currently located.
Format: When there is a change in location, use "from [City A] to [City B]" to denote the transition. If remaining in the same city, simply use the city's name (e.g., "City A").

#### transportation:
Description: Specifies the mode of transportation used.
Format: For flights, include the details in the format "Flight Number: XXX, from [City A] to [City B]". For self-driven or taxi travel, use "self-driving/taxi, from [City A] to [City B]". If there is no travel between cities on that day, use "-".

#### breakfast, lunch, and dinner:
Description: Details about dining arrangements.
Format: Use "Name, City" to specify the chosen restaurant and its location. If a meal is not planned, use "-".

#### attraction:
Description: Information about attractions visited.
Format: List attractions as "Name, City". If visiting multiple attractions, separate them with a semicolon ";". If no attraction is planned, use "-".

Please refer to [this](https://huggingface.co/datasets/osunlp/TravelBench/resolve/main/example_submission.jsonl?download=true) for example submission file. 

Submission made by our team are labelled "TravelBench Team". Each submission will be automatically evaluated and scored based on the predefined metrics. The scores and rankings will be updated and displayed on the leaderboard.

"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@misc{Xie2024TravelBench,
      title={}, 
      author={},
      year={2024},
      eprint={,
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}"""


def format_error(msg):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{msg}</p>"

def format_warning(msg):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{msg}</p>"

def format_log(msg):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{msg}</p>"

def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

