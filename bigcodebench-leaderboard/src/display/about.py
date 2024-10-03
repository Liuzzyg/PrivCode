TITLE = """<div style="text-align: center;"><h1> 🌸<span style='color: #C867B5;'>BigCodeBench</span> Leaderboard</h1></div>\
            <br>\
            <p>Inspired from the <a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard">🤗 Open LLM Leaderboard</a> and <a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">⭐ Big Code Models Leaderboard</a>, we compare performance of LLMs on <a href="https://huggingface.co/datasets/bigcode/bigcodebench">BigCodeBench</a> benchmark.</p>
            <p>To get started, please check out <a href="https://github.com/bigcode-project/bigcodebench">our GitHub repository</a>.
            <br>\
            For more details, please check our <a href="https://huggingface.co/blog/terryyz/bigcodebench-hard">blog on the Hard Set</a>, <a href="https://huggingface.co/blog/leaderboard-bigcodebench">blog on the Full Set</a> and <a href="https://arxiv.org/abs/2406.15877">paper</a>.</p>
            """

ABOUT_TEXT = """# Context
We believe that there are three main expectations of a good execution-based programming benchmark:
1. The benchmark should be easy to use and efficient in evaluating the fundamental capabilities of LLMs. Repo-level and agent-centric benchmarks (e.g., SWE-bench) are not suitable for this purpose.
2. The benchmark should be practical, covering various programming scenarios. Algo-specific benchmarks (e.g., HumanEval and MBPP) are unsuitable. Domain-specific benchmarks (e.g., DS-1000) are also unsuitable for this purpose.
3. The benchmark should be challenging, where the tasks require LLMs' strong compositional reasoning capabilities and instruction-following capabilities. The benchmarks with simple tasks (e.g., ODEX) are unsuitable.

BigCodeBench is the first benchmark that meets all three expectations. It is an <u>*__easy-to-use__*</u> benchmark that evaluates LLMs with <u>*__practical__*</u> and <u>*__challenging__*</u> programming tasks, accompanied by an end-to-end evaluation framework [`bigcodebench`](https://github.com/bigcode-project/bigcodebench). We aim to assess how well LLMs can solve programming tasks in an open-ended setting, with the following two focuses:

- Diverse Function Calls: This design requires LLMs to utilize diverse function calls.
- Complex Instructions: This design requires LLMs to follow complex instructions.


### Benchamrks & Prompts
The dataset has 2 variants: 
1. `BigCodeBench-Complete`: _Code Completion based on the structured long-context docstrings_.
1. `BigCodeBench-Instruct`: _Code Generation based on the NL-oriented instructions_.

Figure below shows the example of `Complete` vs `Instruct` prompt. For `Instruct`, we only focus on instruction-tuned LLMs.

<img src="https://github.com/bigcode-bench/bigcode-bench.github.io/blob/main/asset/bigcodebench_prompt.svg?raw=true" alt="OctoCoder vs Base HumanEval prompt" width="800px">

The specific prompt template can be found [here](https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/model.py).

There are some edge cases:
- Due to the training flaws in StarCoder2 and Granite-Code, we additionally strip the trailing newlines for model inference.
- We have not included the `Instruct` results of Granite-Code-Instruct 8B & 3B as they constantly have empty outputs.

### Evaluation Parameters
- All models were evaluated with the [bigcodebench](https://github.com/bigcode-project/bigcodebench). You can install the [PyPI package](https://pypi.org/project/bigcodebench/).
To get started, please first set up the environment:

```bash
# Install to use bigcodebench.evaluate
pip install bigcodebench --upgrade
# If you want to use the evaluate locally, you need to install the requirements
pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt

# Install to use bigcodebench.generate
# You are strongly recommended to install the generate dependencies in a separate environment
pip install bigcodebench[generate] --upgrade
```

### Scoring and Rankings
- Models are ranked according to Pass@1 using greedy decoding. Setup details can be found <a href="https://github.com/bigcode-project/bigcodebench/blob/main/bigcodebench/generate.py">here</a>.
- The code to compute Elo rating is [here](https://github.com/bigcode-project/bigcodebench/blob/main/analysis/get_results.py), which is based on [Chatbot Arena Notebook](https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR#scrollTo=JdiJbB6pZB1B&line=2&uniqifier=1). We only compute the Elo rating for the `BigCodeBench-Complete` variant.

### Contact
If you have any questions, feel free to reach out to us at [terry.zhuo@monash.edu](mailto:terry.zhuo@monash.edu) or [contact@bigcode-project.org](mailto:contact@bigcode-project.org)

### Citation Information

```bibtex
@article{zhuo2024bigcodebench,
    title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions}, 
    author={Terry Yue Zhuo and Minh Chien Vu and Jenny Chim and Han Hu and Wenhao Yu and Ratnadira Widyasari and Imam Nur Bani Yusuf and Haolan Zhan and Junda He and Indraneil Paul and Simon Brunner and Chen Gong and Thong Hoang and Armel Randy Zebaze and Xiaoheng Hong and Wen-Ding Li and Jean Kaddour and Ming Xu and Zhihan Zhang and Prateek Yadav and Naman Jain and Alex Gu and Zhoujun Cheng and Jiawei Liu and Qian Liu and Zijian Wang and David Lo and Binyuan Hui and Niklas Muennighoff and Daniel Fried and Xiaoning Du and Harm de Vries and Leandro Von Werra},
    journal={arXiv preprint arXiv:2406.15877},
    year={2024}
}
```
"""

SUBMISSION_TEXT = """
<h1 align="center">
How to submit models/results to the leaderboard?
</h1>
We welcome the community to submit evaluation results of new models. We also provide an experimental feature for submitting models that our team will evaluate on the 🤗 cluster.

## Submitting Models (experimental feature)
Inspired from the Open LLM Leaderboard, we welcome code models submission from the community that will be automatically evaluated. Please note that this is still an experimental feature.
Below are some guidlines to follow before submitting your model:

#### 1) Make sure you can load your model and tokenizer using AutoClasses:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
If this step fails, follow the error messages to debug your model before submitting it. It's likely your model has been improperly uploaded.
Note: make sure your model is public!
Note: if your model needs `use_remote_code=True`, we do not support this option yet.
#### 2) Convert your model weights to [safetensors](https://huggingface.co/docs/safetensors/index)
It's a new format for storing weights which is safer and faster to load and use. It will also allow us to add the number of parameters of your model to the `Extended Viewer`!
#### 3) Make sure your model has an open license!
This is a leaderboard for Open LLMs, and we'd love for as many people as possible to know they can use your model 🤗
#### 4) Fill up your model card
When we add extra information about models to the leaderboard, it will be automatically taken from the model card.
"""

SUBMISSION_TEXT_2 = """
## Sumbitting Results
You also have the option for running evaluation yourself and submitting results. These results will be added as non-verified, the authors are however required to upload their generations in case other members want to check.

### 1 - Running Evaluation

We wrote a detailed guide for running the evaluation on your model. You can find the it in [bigcode-evaluation-harness/leaderboard](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/leaderboard). This will generate a json file summarizing the results, in addition to the raw generations and metric files.

### 2- Submitting Results 🚀

To submit your results create a **Pull Request** in the community tab to add them under the [folder](https://huggingface.co/spaces/bigcode/bigcodebench-code-evals/tree/main/community_results) `community_results` in this repository:
- Create a folder called `ORG_MODELNAME_USERNAME` for example `bigcode_my_model_terry`
- Put your json file with grouped scores from the guide, in addition generations folder and metrics folder in it.

The title of the PR should be `[Community Submission] Model: org/model, Username: your_username`, replace org and model with those corresponding to the model you evaluated.
"""

SUBMISSION_TEXT_3 = """
<h1 align="center">
How to submit models/results to the leaderboard?
</h1>
We welcome the community to submit evaluation results of new models. These results will be added as non-verified, the authors are however required to upload their generations in case other members want to check.

### 1 - Running Evaluation

We wrote a detailed guide for running the evaluation on your model. You can find the it in [bigcode-evaluation-harness/leaderboard](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/leaderboard). This will generate a json file summarizing the results, in addition to the raw generations and metric files.

### 2- Submitting Results 🚀

To submit your results create a **Pull Request** in the community tab to add them under the [folder](https://huggingface.co/spaces/bigcode/multilingual-code-evals/tree/main/community_results) `community_results` in this repository:
- Create a folder called `ORG_MODELNAME_USERNAME` for example `bigcode_starcoder_loubnabnl`
- Put your json file with grouped scores from the guide, in addition generations folder and metrics folder in it.

The title of the PR should be `[Community Submission] Model: org/model, Username: your_username`, replace org and model with those corresponding to the model you evaluated.
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"

CITATION_BUTTON_TEXT = r"""
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Minh Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
"""

SUBMISSION_TEXT_3="""
## We welcome the community to submit the evaluation results or request for new models to be added to the leaderboard.
## To submit the evaluation results, please send us your (1) raw generations, (2) sanitized generations, (3) execution logs, and (4) pass rate results to our [email](mailto:terry.zhuo@monash.edu). We will review and add the results to the leaderboard as soon as possible.
## To request for the new model evaluation, please [file an issue](https://github.com/bigcode-project/bigcodebench/issues/new/choose) to add the model to the leaderboard or [start a discussion](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard/discussions/new) in the community 🤗
"""