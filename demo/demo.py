# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0

import logging as log
import transformers
from argparse import ArgumentParser
import  numpy as np
import sys
import torch
from time import perf_counter
import onnx

from launchers import create_launcher, MODEL_TO_URL
from utils import (get_top_k_logits, get_top_p_logits, softmax, stop_criteria, process_logits)

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

MODELS = [
    "GPT-2",
    "GPT-J",
    "GPT-NeoX",
    "BLOOM",
    "OPT"
]

LAUNCHERS = [
    "pytorch",
    "onnx",
    "openvino"
]


def build_argparser():
    parser = ArgumentParser()

    options = parser.add_argument_group('Options')
    options.add_argument('-i', '--input', default="", type=str,
                         help='Optional. An text input to process. If is not specified, use interactive mode')
    options.add_argument('-m', '--model', required=False, choices=MODELS, default="GPT-J",
                         help=f"Optional. Name of using model. Available names = {MODELS}. Default is 'GPT-J'")
    options.add_argument('-l', '--launcher', required=False, choices=LAUNCHERS, default="pytorch",
                         help="Optional. Name of using backend for runtime. Available backends = {LAUNCHERS}. Default is 'PyTorch'")
    options.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    options.add_argument("--top_k", help="Optional. Number of tokens with the highest probability "
                                      "which will be kept for generation",
                        default=5, required=False, type=int)
    options.add_argument("--top_p", help="Optional. Maximum probability, tokens with such a probability "
                                      "and lower will be kept for generation",
                        default=0.7, required=False, type=float)
    options.add_argument("--max_sample_token_num", help="Optional. Maximum number of tokens in generated sample",
                         default=40, required=False, type=int)
    options.add_argument('--dynamic_shape', action='store_true', help='Run model with dynamic input sequence. If not provided, input sequence will be padded to max_seq_len')
    options.add_argument('--max_seq_len', type=int, required=False, default=128, help='Optional. Maximum sequence length for processing. Default value is 128')
    return parser


def create_tokenier_launcher(model_name: str, laucnher_name: str, config):
    start_time = perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TO_URL[model_name])
    launcher = create_launcher(laucnher_name, model_name, config)
    load_time = perf_counter() - start_time

    log.info("Load time = {:.3f} seconds".format(load_time))

    return tokenizer, launcher

def generate_text(tokenizer, launcher, args):
    for prompt in prompts(args.input):
        if not prompt.strip():
            break
        # Get input_ids using tokenizer
        input_ids = tokenizer(prompt, return_tensors="np").input_ids
        eos_token_id = 50256

        cur_input_len = input_ids.shape[-1]
        # maximum number of tokens that will be generated
        max_sample_token_num = args.max_sample_token_num + cur_input_len

        # maximum number of tokens that can be processed by network at once
        max_length = args.max_seq_len

        # Generating process
        t0 = perf_counter()
        t_count = 0
        infer_time = []
        while True:
            model_input = input_ids
            # infer by laucnher
            start_infer = perf_counter()
            if not args.dynamic_shape and args.launcher == "openvino":
                # pad the rest of the request
                pad_len = max_length - cur_input_len
                model_input = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
            outputs = launcher.process(model_input)
            finish_infer = perf_counter()
            infer_time.append(finish_infer - start_infer)
            t_count += 1
            next_token_logits = outputs[:, cur_input_len-1, :]
            # pre-process distribution
            next_token_scores = process_logits(input_ids, next_token_logits, eos_token_id)
            if args.top_k > 0:
                next_token_scores = get_top_k_logits(next_token_scores, args.top_k)
            if args.top_p < 1.0:
                next_token_scores = get_top_p_logits(next_token_scores, args.top_p)
            # get next token id
            probs = softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1, p=probs[0], replace=True)
            # update info for the next step
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
            cur_input_len = input_ids.shape[-1]
            if stop_criteria(input_ids, min(max_sample_token_num, 1024), eos_token_id):
                break

        text = tokenizer.decode(input_ids[0])
        all_time = perf_counter() - t0
        log.info("{} requests were processed in {:0.2f}sec ({:0.2}sec per request, {:0.2}sec per infer)".format(
                 t_count, all_time, all_time / t_count, np.mean(infer_time)))

        # print result
        log.info("GENERATED SEQUENCE: {}".format(text))


def prompts(input_text):
    if input_text:
        log.info("Input prompt: {}".format(input_text))
        yield input_text
    else:
        while True:
            yield input('Type input prompt (empty string to exit):')

def main():
    args = build_argparser().parse_args()

    tokenizer, launcher = create_tokenier_launcher(args.model, args.launcher, config=args)

    generate_text(tokenizer, launcher, args)


if __name__ == '__main__':
    sys.exit(main() or 0)
