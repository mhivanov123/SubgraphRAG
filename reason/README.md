# Stage 2: Reasoning

## Installation
The code is tested with `torch 2.4.0+cu121`, `vllm 0.5.5`, `openai 1.50.2` on `Python 3.10.14`. For other used packages, typically all recent versions should work.

## Inference with LLMs
First put the retriver's results in `./scored_triples`, where we have provided our results for the sake of demonstration. You may need to install Git LFS to download the files due to their sizes.

Then, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
```

Our used config for each dataset can be found in `./config`.

## Our results

We provide our generated predictions in `./results`. Feel free to take a look at them, but you would need to install Git LFS first.
