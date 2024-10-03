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

```
pip install -r requirements_vllm.txt
```

### bigcodebench

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
The script is to finetune the smaller model with dp-sgd. You can set 'MODEL_PATH' to change task scenarios.

## generate private syndata
```
conda activate vllm
```

```
sh data/private_syn/run_generate.sh
```
The script is to generate privata synthetic magicoder data. You can set '--model_path' and '--step' to choose the proper checkpoint.

```
sh data/private_syn/run_clean_data.sh
```
The script is to clean the generated private syndata.

```
sh data/private_syn/run_match_original_data.sh
```
The script is to save the prompt-matched samples in original magicoder oss instruction data.

## finetune stage2
```
conda activate dpcode
```

```
sh run_finetune_dp_step2.sh
```
You can set 'MODEL_PATH_STEP1' to the model name in stage1, 'MODEL_PATH_STEP1' to the model name in stage2.
Commit line 30-31 for finetuning on private syndata, while committing line 26-27 for finetuning on original data.


## evaluate on evalplus
```
python run_evalplus.py
```

Set "is_post_step = True" for evaluating checkpoint saved in dp-finetuning stage2. 
Set "is_private_syndata_step2s = ['yes', 'no']", 'yes' for evaluating model finetuned on private syndata, while 'no' for evaluating model finetuned on original data.
Set "is_pretrained = True" for evaluating pretraining model locally.