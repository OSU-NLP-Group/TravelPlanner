from tqdm import tqdm
import argparse
from openai_request import build_plan_format_conversion_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

def load_local_model(model_id):
    hf_token = os.environ.get("HF_API_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        do_sample=False,
        return_full_text=False,
        temperature=None,
        top_p=None,
        batch_size=1
    )
    return pipe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")
    parser.add_argument("--hf_model_id", type=str, default="/datasets/ai/qwen3/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    args = parser.parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)

    pipe = load_local_model(args.hf_model_id)

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'

    data = build_plan_format_conversion_prompt(
        directory=args.output_dir, set_type=args.set_type,
        model_name=args.model_name, strategy=args.strategy, mode=args.mode
    )
    output_file = f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt'

    for idx, prompt in enumerate(tqdm(data)):
        if prompt == "":
            with open(output_file, 'a+', encoding='utf-8') as f:
                f.write(str(idx) + '\n')
            continue
        result = pipe(prompt + " /no_think")[0]['generated_text']
        with open(output_file, 'a+', encoding='utf-8') as f:
            result_clean = str(idx) + "\t" + "\t".join(result.split("\n"))
            f.write(result_clean + '\n')

    print("Done")