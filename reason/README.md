# Stage 2: Reasoning

## Inference with LLMs
First put the retriver's results in `./scored_triples`, where we have provided our results for the sake of demonstration. You may need to install Git LFS to download the files due to their sizes.

Then, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
```

Our used config for each dataset can be found in `./config`.

## Our results

We provide our generated predictions in `./results`. Feel free to take a look at them, but you would need to install Git LFS first.
