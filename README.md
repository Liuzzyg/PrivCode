<div align=center>
  
# PrivCode: When Code Synthesis Meets Differential Privacy
</div>


This is the official implementaion of paper ***PrivCode: When Code Synthesis Meets Differential Privacy***. This repository contains Pytorch training code and evaluation code. PRIVCODE is a Differetial Privacy (DP) code synthesis tool, which leverages the DP technique to generate synthetic code, allowing organizations to share and utilize code LLMs without privacy concerns.



## 1. Contents
- PrivCode: When Code Synthesis Meets Differential Privacy
  - [1. Contents](#1-contents)
  - [2. Project structure](#2-project-structure)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset](#32-dataset)
  - [4. Running Instructions](#running-instructions)
  - [5. Acknowledgment](#5-acknowledgment)


## 2. Project structure

The structure of this project is as follows:
```
data
  -- private_syn -------------------------------- the scripts of data generation and post-process filter in utility experiment
  -- pii_dataset -------------------------------- the scripts of data generation and post-process filter in PII protection experiment
  -- vulnerability ------------------------------ the scripts of data generation and post-process filter in vulnerability protection experiment
eval_bcb
  -- run_bigcodebench_step2.sh ------------------ evaluate the utility of PrivCode on BigCodeBench benchmark in utility experiment
  -- run_bigcodebench_dpbaseline_step2.sh ------- evaluate the utility of DPFT on BigCodeBench benchmark in utility experiment
  -- run_bigcodebench_infbaseline_step2.sh ------ evaluate the utility of NonDPFT on BigCodeBench benchmark in utility experiment
  -- run_bigcodebench_pretrain.sh --------------- evaluate the utility of PreCode on BigCodeBench benchmark in utility experiment
eval_evalplus
  -- run_evalplus_0.3.1_step2.sh ---------------- evaluate the utility of PrivCode and NonDPFT on EvalPlus benchmark in utility experiment
  -- run_evalplus_0.3.1_dpbaseline.sh ----------- evaluate the utility of DPFT on EvalPlus benchmark in utility experiment
  -- run_bigcodebench_pretrain.sh --------------- evaluate the utility of PreCode on EvalPlus benchmark in utility experiment
  -- run_evalplus_0.3.1_step2_ablation.sh ------- evaluate the utility of variants of PrivCode on EvalPlus benchmark in ablation experiment
  -- run_evalplus_0.3.1_step2_hyper.sh ---------- evaluate the utility of PrivCode on EvalPlus benchmark in hyper-parameter analysis experiment
-- examples ------------------------------------- util scripts and config files of training
-- fastDP --------------------------------------- differential privacy finetuning engine
pii_leaks_eval
-- detector.py ---------------------------------- PII protection evaluation benchmark
-- prompt_template.py --------------------------- template for prompts that trigger the reproduction
 of PIIs
-- run_pii_detect_step2.sh ---------------------- evaluate PrivCode’s ability to protect PIIs
-- run_pii_detect_step2_infbaseline.sh ---------- evaluate NonDPFT’s ability to protect PIIs

```











## Installation

### dpcode
Install the required dependencies:

```
pip install -r requirements_dpcode.txt
```

Install human-eval

```
pip install -e human-eval
```

Install human-eval

```
cd evalplus
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -r requirements.txt
```
### vllm
Install dependencies for vllm:
```
pip install -r requirements_vllm.txt
```

### bigcodebench
To set up bigcodebench, follow these steps(https://github.com/bigcode-project/bigcodebench):
```
cd bigcodebench
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Install to use bigcodebench.evaluate
pip install -e .
# Install to use bigcodebench.generate
pip install -e .[generate]
# https://github.com/huggingface/alignment-handbook/issues/180
pip install deepspeed==0.14.4
```

## dp-finetune stage1
```
conda activate dpcode
```

```
sh run_finetune_dp_step1.sh
```
This script fine-tunes a smaller model using DP-SGD. You can modify 'MODEL_PATH' to switch between different task scenarios.

## generate private syndata
```
conda activate vllm
```

```
sh data/private_syn/run_generate.sh
```
In this script, you can adjust the --model_path and --step arguments to select the desired checkpoint.

To clean the generated data, run:
```
sh data/private_syn/run_clean_data.sh
```

Finally, to match the generated data with the original Magicoder OSS instruction data, use:
```
sh data/private_syn/run_match_original_data.sh
```

## finetune stage2
```
conda activate dpcode
```

```
sh run_finetune_dp_step2.sh
```
• You can set 'MODEL_PATH_STEP1' to the model name in stage1, 'MODEL_PATH_STEP1' to the model name in stage2.  
  
• Commit line 30-31 for finetuning on private syndata, while committing line 26-27 for finetuning on original data.


## evaluate on evalplus
```
conda activate dpcode
```
```
python run_evalplus.py
```

• Set "is_post_step = True" for evaluating checkpoint saved in dp-finetuning stage2.  
  
• Set "is_private_syndata_step2s = ['yes', 'no']", 'yes' for evaluating model finetuned on private syndata, while 'no' for evaluating model finetuned on original data.  
  
• Set "is_pretrained = True" for evaluating pretraining model locally.


## evaluate on bigcodebench
```
conda activate bigcodebench
```
```
sh run_bigcodebench.sh
```
You can adjust the script as needed by commenting unused command. Or you can leave this to me:)!



## docker exeuate code clean
```
docker run -it --entrypoint /bin/bash code-cleaner-with-bash:latest
```
```
docker cp /data_path container_id:/app
```
```
bash data/private_syn/run_clean_data.sh
```



## round-trip test
Installation
```
pip install -r requirements_round_trip.txt
```

Get start

In the line 112 of "data/private_syn/round_trip_test_prompt.py", change the 'download_dir' to your own cache dir, or commit it.
```
bash data/private_syn/run_rt_test_prompt.sh
```


## docker vulnerability eval
```
docker run --gpus all -it vulnerability_eval
```

```
docker cp /model_path container_id:/app
```

```
docker cp SafeCoder container_id:/app
```

```
bash SafeCoder/scripts/run_sec_eval.sh
```

```
bash SafeCoder/scripts/run_print_results.sh
```

local:
```
bash SafeCoder/experiments/docker_cp.sh
```