import argparse
import json
from tqdm import tqdm
from evaluation.commonsense_constraint import evaluation as commonsense_eval
from evaluation.hard_constraint import evaluation as hard_eval


def load_line_json_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.read().strip().split("\n"):
            unit = json.loads(line)
            data.append(unit)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Check format validity of generated travel plans using placeholder query"
    )
    parser.add_argument(
        "--evaluation_file_path",
        type=str,
        required=True,
        help="Path to the JSONL file containing plans to check",
    )
    args = parser.parse_args()

    data_list = load_line_json_data(args.evaluation_file_path)

    placeholder_query = {
        "org": "Minneapolis",
        "dest": "Ohio",
        "days": 7,
        "visiting_city_number": 3,
        "date": [
            "2022-03-17",
            "2022-03-18",
            "2022-03-19",
            "2022-03-20",
            "2022-03-21",
            "2022-03-22",
            "2022-03-23",
        ],
        "people_number": 2,
        "local_constraint": {
            "house rule": "pets",
            "cuisine": ["American", "Mediterranean", "Chinese", "Italian"],
            "room type": "placeholder",
            "transportation": "no flight",
        },
        "budget": 5100,
        "query": "We require a 7-day travel itinerary for two leaving from Minneapolis and covering three cities in Ohio, starting from March 17th to March 23rd, 2022. Our budget is up to $5,100. We will be accompanied by our pets, so we need pet-friendly accommodations. Our meals should preferably include American, Mediterranean, Chinese, and Italian cuisines. Please note we prefer not to take any flights so our travel plan should not include them.",
        "level": "hard",
    }

    for idx in tqdm(range(len(data_list))):
        tested_plan = data_list[idx]
        if tested_plan["plan"]:
            commonsense_info_box = commonsense_eval(placeholder_query, tested_plan["plan"])
        else:
            commonsense_info_box = None

        if (
            commonsense_info_box
            and commonsense_info_box["is_not_absent"][0]
            and commonsense_info_box["is_valid_information_in_sandbox"][0]
        ):
            hard_info_box = hard_eval(placeholder_query, tested_plan["plan"])


if __name__ == "__main__":
    main()
