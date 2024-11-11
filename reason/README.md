# Stage 2: Reasoning

## Installation
The code is tested with `torch 2.4.0+cu121`, `vllm 0.5.5`, `openai 1.50.2` on `Python 3.10.14`. For other used packages, typically all recent versions should work.

## Our results
We provide our retriever's results in `./scored_triples` and our reasoning results in `./results`. Please run the following command to download all our results.

```
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./
```


## Inference with LLMs
First put the retriver's results in `./scored_triples`. Then, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
```

Our used config for each dataset can be found in `./config`.

