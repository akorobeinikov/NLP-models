# Repo for inference of NLP models in different fremeworks

## Hardware
For inference was used server with following parameters:
* CPU: 2 x Intel® Xeon® Gold 6248R
* RAM: 376 GB

## Models

| Model name        | Implementation   | Model card                         |
|-------------------|------------------|------------------------------------|
| GPT-2             | PyTorch          | [gpt-2](models/GPT-2/README.md)    |
| GPT-J             | PyTorch          | [gpt-j](models/GPT-J/README.md)    |
| GPT-NeoX          | PyTorch          | [gpt-neox](models/GPT-J/README.md) |

## Prerequisites

Before running inference, create python environment:
```
  python3 -m venv <env_name>
  source <env_name>/bin/activate
```

And install the necessary dependencies:
```
  python3 -m pip install -r requirements.txt
```

## Experiments

Results of performance experiments:

| Model name        | Number of parameters   | Size of model | PyTorch time  | ONNXRuntime time | OpenVINO time |
|-------------------|------------------------|---------------|---------------|------------------|---------------|
| GPT-2             | 175M                   | 240-624M      | 3.5 - 3.7 sec | 0.5 sec          | 1.3 - 1.7 sec |
| GPT-J             | 6B                     | 24G           | 34 sec        | 8 - 10 sec       | 10 - 13 sec   |
| GPT-NeoX          | 6B                     | 39G           | 90 sec        | 32 - 35 sec      | 50 sec        |
