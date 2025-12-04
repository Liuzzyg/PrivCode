<div align=center>
  
# PrivCode: When Code Synthesis Meets Differential Privacy
</div>


This is the official implementation of the paper ***PrivCode: When Code Synthesis Meets Differential Privacy***. This repository provides PyTorch-based training and evaluation code. PrivCode is a Differential Privacy (DP) code synthesis framework that leverages DP techniques to generate synthetic code, enabling organizations to share and utilize code LLMs without compromising privacy.

<div align=center>
<img src="./images/figures_overview_00.png" width = "1000" alt="The workflow of PrivCode" align=center />
</div>

## 1. Contents

- PrivCode: When Code Synthesis Meets Differential Privacy
  - [1. Contents](#1-contents)
  - [2. Project structure](#2-project-structure)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset](#32-dataset)
  - [4. Running Instructions](#4-running-instructions)
  - [5. Computational Resource Requirements](#5-Computational-Resource-Requirements)
  - [6. Acknowledgment](#6-acknowledgment)

## 2. Project structure

The structure of this project is as follows:
```
canary
  -- eval-leakage-rate -------------------------- evaluate the leakage rate of canaries
  -- origin_data -------------------------------- canary data and scripts of yielding canary-injected datasets
  -- scripts-run-finetune ----------------------- scripts of PrivCode and baselines in canary experiments
data
  -- private_syn -------------------------------- the scripts of data generation and post-process filter of utility-boosting stage in utility experiment
  -- pii_dataset -------------------------------- the scripts of yielding OSS-Instruct PII dataset, data generation and post-process filter of utility-boosting stage in PII protection experiment
eval_utility
  -- eval_bcb ----------------------------------- evaluate the utility on BigCodeBench benchmark in utility experiment
  -- eval_evalplus ------------------------------ evaluate the utility on EvalPlus benchmark in utility experiment
examples ---------------------------------------- training util scripts and config files of training
fastDP ------------------------------------------ differential privacy finetuning engine
scripts-run-finetune
  -- privcode_privacy_sanitizing.sh ------------- script of privacy-sanitizing stage of PrivCode
  -- privcode_utility_boosting.sh --------------- script of utility-boosting stage of PrivCode
scripts-run-merge-peft -------------------------- script of merge peft model to base model

```


## 3. Get Start

### 3.1 Installation

To install, clone the repository and run the following:

```bash 
# install privcode environment for training (fine-tuning steps)
conda create -n privcode python==3.11.0
conda activate privcode
pip install -r requirements_privcode.txt

# install privcode_infer environment for inference (data synthesis, round-trip validation, and evaluation steps)
conda create -n privcode_infer python==3.11.0
conda activate privcode_infer
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
pip install bigcodebench --upgrade
pip install packaging ninja
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install bert_score

# if your system misses ccrypt dependency 
pip install ccrypt
```

The code was tested on Python 3.11.

### 3.2 Dataset

In the utility experiments, we use the dataset [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) released on huggingface.

In the canary experiments, we conduct and release the [OSS-Instruct PII dataset](https://huggingface.co/datasets/bigcode/bigcode-pii-dataset). based on [bigcode/bigcode-pii-dataset](https://huggingface.co/datasets/bigcode/bigcode-pii-dataset).

### 3.3 Pre-train Models

Our selection of junior and premium LLM models are as follows, they will be downloaded from huggingface when you run the fine-tuning scripts.

| **Model Type**    | **LLM Model**                                                                 |
|--------------------|-------------------------------------------------------------------------------|
| **Junior Model**   | [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)  |
| **Premium Model**  | [deepseek-ai/deepseek-coder-6.7b-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) <br> [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) <br> [google/codegemma-7b](https://huggingface.co/google/codegemma-7b) <br> [Qwen/CodeQwen1.5-7B](https://huggingface.co/Qwen/CodeQwen1.5-7B) |

## 4. Running Instructions
Overview: [4.1 Utility Experiment](#41-Implementations-for-Utility-Experiment-Results-in-Table-3) introduces the implementations for the utility experiment results in Table 3. Next, [4.2 Canary Experiment](#42-Implementations-for-Canary-Experiment-Results-in-Table-4) introduces the implementations for the canary experiment results in Table 4. Finally, [4.3 Hyper paramter Anlysis](#43-Hyper-paramter-Anlysis-Implementations-for-Results-in-Table-5) introduces the implementations for the hyper-paramter anlysis results in Table 5 and Figure 5.

We list the key hyper-parameters below, including their explanations,

- `model_path`: the name of pre-training LLM, or the path of checkpoint.
- `dataset_name`: the name of fine-tuning dataset.
- `max_global_steps`: the max global training step, actual steps divided by gradient accumulation steps.
- `alpha`: controls how quickly the KL loss weight decays over training steps (decay rate).
- `max lambda`: the initial maximum weight of the KL loss term (strongest influence in early training).
- `min lambda`: the final minimum weight of the KL loss term (lowest influence in later training).
- `target epsilon`: controls the privacy budget.
- `rt model`: the name of the round-trip LLM.
- `sim threshold`: the threshold to filter out synthetic data with low round-trip correctness.
- `output dir`: the path of saved model configs and checkpoints.


### 4.1 Implementations for Utility Experiment Results in Table 3.

#### Step1: Privacy-sanitizing Stage of PrivCode.

For fine-tuning with Privacy-free Syntax-Aware (PrivSA) module, run:
```bash
conda activate privcode
bash scripts-run-finetune/privcode_privacy_sanitizing.sh
python scripts-run-merge-peft/privcode_privacy_sanitizing.py
```

> **Note:** You can control the privacy budget by adjust the "TARGET_EPSILONs".

#### Step2: Utility-boosting Stage of PrivCode.

To generate privacy-free data, run:
```bash 
conda activate privcode_infer
bash data/private_syn/run_generate.sh
```
Example saving path of the generated data: ```"data/private_syn/Qwen2.5-Coder-1.5B/dp4_lambda1000to0.1_alpha0.01.jsonl"```.

For Execution filter, run:
```bash 
docker pull liuzhengyg/privcode-execution-filter:latest
docker run -it --entrypoint /bin/bash liuzhengyg/privcode-execution-filter:latest
docker cp data/private_syn/Qwen2.5-Coder-1.5B/dp4_lambda1000to0.1_alpha0.01.jsonl container_id:/app/data/private_syn/Qwen2.5-Coder-1.5B # copy your local data file to the container in another terminal
bash data/private_syn/run_clean_data.sh
docker cp container_id:/app/data/private_syn/Qwen2.5-Coder-1.5B/cleaned_dp4_lambda1000to0.1_alpha0.01.jsonl data/private_syn/Qwen2.5-Coder-1.5B  # copy the execution-filtered data file to the local in another terminal
```
> **Note:** Ensure you have installed docker.

For round-trip filter, run:
```bash 
conda activate privcode_infer
bash data/private_syn/run_rt_test_prompt.sh
```
Example saving path of the generated data: ```"data/private_syn/Qwen2.5-Coder-1.5B/Llama-3.1-8B-Instruct_tau0.88/final_dp4_lambda1000to0.1_alpha0.01.jsonl"```.

For fine-tuning without DP, run:

```bash 
conda activate privcode
bash scripts-run-finetune/privcode_utility_boosting.sh
python scripts-run-merge-peft/privcode_utility_boosting.py
```

#### Step3: Utility Evaluation of PrivCode.

Compute the pass@1 rate in EvalPlus benchmark:

```bash 
conda activate privcode_infer
bash eval-utility/eval-evalplus/run_evalplus_privcode.sh
```

A result of PrivCode HumanEval-Instruct of Qwen2.5-Coder-7B in Table 3 is as follows, "base_tests" means HumanEval pass@1 score, while "plus_tests" means HumanEval+ pass@1 score.
```
{"dataset": "humaneval", "results": {"base_tests": {"pass@1": 0.6646341463414634}, "plus_tests": {"pass@1": 0.6097560975609756}}}
```

Compute the pass@1 rate in BigCodeBench benchmark:

```bash 
conda activate privcode_infer
bash eval-utility/eval-bcb/run_bigcodebench_privcode.sh
```

A result of PrivCode BigCodeBench-Instruct-Full of Qwen2.5-Coder-7B in Table 3 is as follows,
```
{"pass@1": 0.2293859649122807, ...}
```


### 4.2 Implementations for Canary Experiment Results in Table 4.

#### Training of PrivCode:

The training steps refers to [4.1 Utility Experiment](#41-Implementations-for-Utility-Experiment-Results-in-Table-3). Here we use canary-injected OSS-Instruct PII datasets as sensitive datasets, you can obtain the raw [OSS-Instruct PII datasets](https://huggingface.co/datasets/ZhengLiu33/OSS-Instruct-PII-dataset) and generate them by running the script [```inject.py```](canary/origin_data/inject.py).

#### Evaluation:

Compute the canary leakage rate:
```bash 
conda activate privcode_infer
bash canary/eval-leakage-rate/run_pii_detect_step2_infbaseline.sh
```
Results of PrivCode under epsilon=4 and without DP protection of Qwen2.5-Coder-7B in Table 4 are as follows,
```
# epsilon=inf
{"canary_counts": {"sarah.connor1990@gmail.com": 1, "Liam Alexander Brooks": 4, "10.0.0.138": 19, "Tr0ub4dor&3xplor3r": 19, "SkyWalker42": 0}, "leakage_rate_percent": 80.0, "unique_canaries_detected": ["Tr0ub4dor&3xplor3r", "10.0.0.138", "Liam Alexander Brooks", "sarah.connor1990@gmail.com"]}
# epsilon=4
{"canary_counts": {"sarah.connor1990@gmail.com": 0, "Liam Alexander Brooks": 0, "10.0.0.138": 0, "Tr0ub4dor&3xplor3r": 0, "SkyWalker42": 0}, "leakage_rate_percent": 0.0, "unique_canaries_detected": []}
```

### 4.3 Hyper-paramter Anlysis (Implementations for Results in Table 5)

For fine-tuning under variant hyper-parameters, run:
```bash 
conda activate privcode
bash scripts-run-finetune/privcode_utility_boosting_hyper.sh
```
> **Note:** You can adjust the max lambda $\lambda_{\text{max}}$, privacy budget $\epsilon$ and BERTScore threshold $\tau_{\text{s}}$ by setting the ```MAX_LAMBDA```, ```TARGET_EPSILON```, ```SIM_THRESHOLD```.

## 5. Computational Resource Requirements

All methods are implemented on a server equipped with four RTX 6000 Ada Generation GPUs and 48GB of memory. We list the GPU memory consumption and runtime of PrivCode and baselines

| Method   | Stage                             | GPU Memory | Training Time |
|-----------|----------------------------------|-------------|----------------|
| **PrivCode** | Privacy-sanitizing Fine-tuning  | 24.4 GB     | 2.37 h         |
|           | Privacy-free Data Synthesis       | 15.8 GB     | 0.39 h         |
|           | Execution Validation              | –           | 0.25 h         |
|           | Round-trip Validation             | 78.7 GB     | 0.45 h         |
|           | Utility-boosting Fine-tuning      | 32.4 GB     | 1.13 h         |
| **JFT**   | The First Non-DP Fine-tuning      | 29.7 GB     | 2.11 h         |
|           | The Second DP Fine-tuning         | 37.3 GB     | 1.74 h         |
| **DPFT**  | DP Fine-tuning Stage              | 38.2 GB     | 2.08 h         |


## 6. Acknowledgements

PrivCode builds upon many works and open-source codebases in both open-source LLMs and benchmarks. We would like to particularly thank the authors of [Fast Differential Privacy](https://github.com/awslabs/fast-differential-privacy), [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder), [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder), [codegemma](https://huggingface.co/google/codegemma-7b), [CodeQwen1.5](https://qwenlm.github.io/blog/codeqwen1.5/), [EvalPlus](https://github.com/evalplus/evalplus), [BigCodeBench](https://github.com/bigcode-project/bigcodebench), [pii-dataset](https://huggingface.co/datasets/bigcode/bigcode-pii-dataset), [SafeCoder](https://github.com/eth-sri/SafeCoder). We sincerely thank them for their contributions to the community.
