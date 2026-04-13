from tqdm import tqdm
import argparse
from .openai_request import build_plan_format_conversion_prompt, prompt_chatgpt


def main():
    parser = argparse.ArgumentParser(
        description="Parse generated travel plans using GPT-4 to convert natural language to structured JSON format"
    )
    parser.add_argument(
        "--set_type",
        type=str,
        default="validation",
        help="Dataset split to process: train, validation, or test",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo-1106",
        help="Model name used to generate the plans",
    )
    parser.add_argument(
        "--mode", type=str, default="two-stage", help="Planning mode: two-stage or sole-planning"
    )
    parser.add_argument(
        "--strategy", type=str, default="direct", help="Strategy used (for sole-planning mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory containing generated plan JSON files",
    )
    parser.add_argument(
        "--tmp_dir", type=str, default="./", help="Directory to save parsed output text file"
    )

    args = parser.parse_args()

    if args.mode == "two-stage":
        suffix = ""
    elif args.mode == "sole-planning":
        suffix = f"_{args.strategy}"
    data = build_plan_format_conversion_prompt(
        directory=args.output_dir,
        set_type=args.set_type,
        model_name=args.model_name,
        strategy=args.strategy,
        mode=args.mode,
    )
    output_file = f"{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt"

    total_price = 0
    for idx, prompt in enumerate(tqdm(data)):
        if prompt == "":
            with open(output_file, "a+", encoding="utf-8") as f:
                assistant_output = str(idx)
                f.write(assistant_output + "\n")
            continue
        results, _, price = prompt_chatgpt(
            "You are a helpful assistant.",
            index=idx,
            save_path=output_file,
            user_input=prompt,
            model_name="gpt-4-1106-preview",
            temperature=0,
        )
        total_price += price

    print(f"Parsing Cost:${total_price}")


if __name__ == "__main__":
    main()
