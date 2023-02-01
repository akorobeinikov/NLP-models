# GPT-NeoX

## Use Case and High-Level Description

GPT-NeoX-20B is a 20 billion parameter autoregressive language model trained on the [Pile](https://arxiv.org/abs/2101.00027). Technical details about GPT-NeoX-20B can be found in the [associated paper](https://arxiv.org/abs/2204.06745). The configuration file for this model is both available at ./configs/20B.yml and included in the download links below.


## Specification

| Metric            | Value            |
|-------------------|------------------|
| Type              | Text Generation  |
| GFlops            | -                |
| number of layers  | 44               |
| Params            | 20 B             |
| Embedding size    | 4096             |
| Vocab size        | 50400            |
| Source framework  | PyTorch\*        |

## Input

### Original model

Token ids, name: `input`, dynamic shape in the format `B, L`, where:

- `B` - batch size
- `L` - sequence length

### Converted model

Token ids, name: `input`, dynamic shape in the format `B, L`, where:

- `B` - batch size
- `L` - sequence length

## Output

### Original model

Prediction scores of language modeling head, name: `output`, dynamic shape `B, L, 50257` in the format `B, L, S`, where:

- `B` - batch size
- `L` - sequence length
- `S` - vocab size

### Converted model

Prediction scores of language modeling head, name: `output`, dynamic shape `B, L, 50257` in the format `B, L, S`, where:

- `B` - batch size
- `L` - sequence length
- `S` - vocab size


## Demo usage

To run demo with GPT-NeoX, use the next command line:

```
  python3 demo.py -m GPT-NeoX -l <launcher_name>
```

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/huggingface/transformers/master/LICENSE).
