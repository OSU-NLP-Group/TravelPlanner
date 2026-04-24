from tqdm import tqdm
import argparse
from openai_request import build_plan_format_conversion_prompt
import os
from langchain_openai import AzureChatOpenAI

def load_azure_model():
    llm = AzureChatOpenAI(
        max_tokens=4096,
        azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2025-01-01-preview'),
    )
    return llm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-5.4-mini")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")
    args = parser.parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)

    llm = load_azure_model()

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'

    data = build_plan_format_conversion_prompt(
        directory=args.output_dir, set_type=args.set_type,
        model_name=args.model_name, strategy=args.strategy, mode=args.mode
    )
    output_file = f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt'

    import time
    for idx, prompt in enumerate(tqdm(data)):
        if prompt == "":
            with open(output_file, 'a+', encoding='utf-8') as f:
                f.write(str(idx) + '\n')
            continue
        time.sleep(3)  # rate limit buffer
        result = llm.invoke(prompt)
        content = result.content
        if not content and hasattr(result, 'additional_kwargs'):
            content = result.additional_kwargs.get('reasoning_content', '')
        if not content:
            content = str(result)
        with open(output_file, 'a+', encoding='utf-8') as f:
            result_clean = str(idx) + "\t" + "\t".join(content.split("\n"))
            f.write(result_clean + '\n')

    print("Done")