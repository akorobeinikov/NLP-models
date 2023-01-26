# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0

import logging as log
import transformers
from argparse import ArgumentParser
import sys
import torch
from torchsummary import summary
from time import perf_counter
import onnx
from pathlib import Path

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
    options.add_argument('-m', '--model', required=False, choices=MODELS, default="GPT-J",
                         help=f"Optional. Name of using model. Available names = {MODELS}. Default is 'GPT-J'")
    options.add_argument('-o', '--output', required=True,
                         help='Optional. Name of the output file(s) to save.')
    return parser


@torch.no_grad()
def convert_to_onnx(model, input_shapes, output_file):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    # output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_inputs = tuple(
        torch.zeros(input_shape, dtype=torch.long)
        for input_shape in input_shapes)
    model(*dummy_inputs)
    torch.onnx.export(model, dummy_inputs, str(output_file), verbose=False, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size", 1: "sequence_len"},"output": {0: "batch_size", 1: "sequence_len"}})

    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)


def load_model(model_name):
    start_time = perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TO_URL[model_name])
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_TO_URL[model_name], torch_dtype=torch.float32)
    # print(model)
    # summary(model, (1, 1024), dtypes=[torch.long])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_time = perf_counter() - start_time

    log.info("Load time = {:.3f} seconds".format(load_time))

    return tokenizer, model, device

def main():
    args = build_argparser().parse_args()

    tokenizer, model, device = load_model(args.model)

    convert_to_onnx(model, input_shapes=[[1,1024]], output_file=Path(args.output))


if __name__ == '__main__':
    sys.exit(main() or 0)
