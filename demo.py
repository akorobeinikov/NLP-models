# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0

import transformers
from argparse import ArgumentParser
import sys
import torch


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
    options.add_argument('-i', '--input', required=True, type=str,
                         help='Required. An text input to process.')
    options.add_argument('-m', '--model', required=False, choices=MODELS, default="GPT-J",
                         help=f"Optional. Name of using model. Available names = {MODELS}. Default is 'GPT-J'")
    options.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    return parser


def generate_text(input, tokenizer, model, device="cpu"):
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)

    generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)
    generated_text = tokenizer.decode(generated_ids[0])

    return generated_text


def main():
    args = build_argparser().parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TO_URL[args.model])
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_TO_URL[args.model], torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    result = generate_text(args.input, tokenizer, model, device)
    print(result)


if __name__ == '__main__':
    sys.exit(main() or 0)
