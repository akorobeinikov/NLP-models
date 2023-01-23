# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0

import logging as log
import transformers
from argparse import ArgumentParser
import sys
import torch
from time import perf_counter

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

MODELS = [
    "GPT-J",
    "GPT-NeoX",
    "BLOOM",
    "OPT"
]

MODEL_TO_URL = {
    "GPT-J": "EleutherAI/gpt-j-6B",
    "GPT-NeoX": "EleutherAI/gpt-neox-20b",
    "BLOOM": "bigscience/bloom",
    "OPT": "facebook/opt-66b"
}

def build_argparser():
    parser = ArgumentParser()

    options = parser.add_argument_group('Options')
    options.add_argument('-i', '--input', default="", type=str,
                         help='Optional. An text input to process. If is not specified, use interactive mode')
    options.add_argument('-m', '--model', required=False, choices=MODELS, default="GPT-J",
                         help=f"Optional. Name of using model. Available names = {MODELS}. Default is 'GPT-J'")
    options.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    return parser


def load_model(model_name):
    start_time = perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TO_URL[model_name])
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_TO_URL[model_name], torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_time = perf_counter() - start_time

    log.info("Load time = {:.3f} seconds".format(load_time))

    return tokenizer, model, device

def generate_text(input, tokenizer, model, device="cpu"):
    start_time = perf_counter()
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)

    generated_ids = model.generate(input_ids, do_sample=True, temperature=0.8, max_new_tokens=100)
    generated_text = tokenizer.decode(generated_ids[0])
    infer_time = perf_counter() - start_time

    return (generated_text, infer_time)


def interactive_mode(tokenizer, model, device="cpu"):
    user_input = input("Please enter new input: ")
    all_time = 0
    count_inputs = 0
    while(user_input!="exit"):
        model_answer, time = generate_text(user_input, tokenizer, model, device)
        all_time += time
        count_inputs += 1
        print(f"Answer:  {model_answer}")
        user_input = input("Please enter new input: ")

    infer_time = all_time / count_inputs
    log.info("Average infer time = {:.3f} seconds".format(infer_time))
    return


def main():
    args = build_argparser().parse_args()

    tokenizer, model, device = load_model(args.model)

    if args.input:
        model_answer, time = generate_text(args.input, tokenizer, model, device)
        print(f"Answer:  {model_answer}")
        log.info("Average infer time = {:.3f} seconds".format(time))
    else:
        interactive_mode(tokenizer, model, device)


if __name__ == '__main__':
    sys.exit(main() or 0)
