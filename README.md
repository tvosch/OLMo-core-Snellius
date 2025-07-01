# OLMo-core on Snellius
This codebase helps Snellius users to quickly set up their LLM pretraining tasks using [OLMo-core](https://github.com/allenai/OLMo-core/tree/main) codebase


A few Snellius-specific pointers:
- GPU: NVIDIA H100 SXM5 94GB (4 GPUs per node)
- Interconnect: Infiniband HDR200
- Persistent storage is provided upon grant agreement under /projects/0/prjsXXXX
- Temporary storage (/scratch-shared/$USER/) for model checkpointing and log saving
- Operating system: Red Hat Enterprise Linux 9.4 (Plow)



## Installation

1. Clone this repository
```bash
git clone git@github.com:tvosch/OLMo-core-Snellius.git
```

2. Set up Snellius environment
```
module load 2024 Python/3.12.3-GCCcore-13.3.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
uv sync # installs to .venv in current directory
uv sync --group flash-attention --no-build-isolation # if hardware support flash attention 2
```

In case uv is not installed, please either `pip install uv` or install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/)
## Dataset
The codebase has been tested on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 10 billion tokens

```python
import os
from datasets import load_dataset

project_space = os.environ.get("PROJECT_SPACE", os.getcwd())
cache_dir = os.path.join(project_space, "my_hf_cache_dir")
output_path = os.path.join(project_space, "datasets", "FineWeb", "fineweb-10BT.jsonl")

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

shard = "sample-10BT"
dataset = load_dataset("HuggingFaceFW/fineweb", shard, cache_dir=cache_dir, split="train")
dataset.to_json(output_path)
```

### Tokenization/Preprocessing
- Estimated time: 1 hour

OLMo-core assumes the data to be pretokenized and in a specific format. For FineWeb, we can achieve this by running:

Assuming a CPU or GPU node is allocated via salloc:
```bash
python pretokenize.py \
        --input_json <path_to_fineweb-10BT.jsonl> \
        --output_dir <path_to_output> \
        --tokenizer meta-llama/Llama-3.1-8B \
        --eos 128001 \
        --num_procs 128 \
```

The output is a directory which memory-mapped numpy (.npy) files containing tokenized samples

## Pre-training
- Estimated time: 10 minutes to 5 days

### Sbatch
Via sbatch is preferred for a longer runs and is called like:
```bash
sbatch train.job
```



## Acknowledgments
Thanks to the [OLMo-core](https://github.com/allenai/OLMo-core/tree/main) team for establishing this framework!
