### Installation
## dpcode
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
## vllm

```
pip install -r requirements_vllm.txt
```

## bigcodebench

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

### dp-finetune stage1
```
conda activate dpcode
sh run_finetune_dp_step1.sh
```

### generate private syndata
```
conda activate vllm
```

```
sh data/private_syn/run_generate.sh
sh data/private_syn/run_clean_data.sh
sh data/private_syn/run_match_original_data.sh
```

### finetune stage2
```
conda activate dpcode
sh run_finetune_dp_step1.sh
```
