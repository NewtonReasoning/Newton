<div style="text-align: center;">
<h1> NEWTON: Are Large Language Models Capable of Physical Reasoning?</h1>

[Yi Ru Wang](https://helen9975.github.io/)$^1$, [Jiafei Duan](https://duanjiafei.com/)$^1$,  [Dieter Fox](https://homes.cs.washington.edu/~fox/)$^{1,2}$, [Siddhartha Srinivasa](https://goodrobot.ai/)$^1$

$^1$ University of Washington, $^2$ NVIDIA

[Project Page](https://newtonreasoning.github.io/) | [Arxiv](https://arxiv.org/abs/2310.07018) | [HuggingFace API (Coming Soon)]()

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="imgs/teaser.gif">

🌟 **NEWTON: Evaluating Large Language Models for Physics Reasoning** 🌟

Are you curious about the physical reasoning abilities of Large Language Models (LLMs) like GPT-4 in different contexualized settings? Look no further! NEWTON is here to help.

🚀 **What is NEWTON?** 🚀

NEWTON is a repository and benchmark designed to assess the physics reasoning skills of LLMs. While these models excel in many language tasks, their grasp of physical concepts often remains unexplored.

🔬 **What's Inside NEWTON?** 🔬

* **Repository**: We provide a collection of 2800 object-attribute pairs, serving as a foundation for generating customizable assessment templates tailored to your specific needs.
* **Benchmark**: We've curated 160k QA questions to evaluate LLMs across foundational, explicit, and implicit physics reasoning tasks. Discover how these models perform in scenarios involving everyday objects and attributes.
* **Pipeline**: A pipeline to synthesize evaluation sets tailored to particular applications.

🤖 **Real-World Applications** 🤖

NEWTON's potential extends beyond evaluation. It can pave the way for integrating LLMs into physically grounded settings, such as robotic manipulation.

❓ If you have any questions, please contact [me](https://helen9975.github.io/) at `yiruwang [at] cs [dot] washington [dot] edu`. ❓

## :open_file_folder: Repository Structure 
<details open>
<summary>[Click to view]</summary>

```
Newton/
│   README.md
|   .gitignore
|   LICENSE
│   gpt_track1.py -- Inference using GPT on Track 1
│   gpt_track2.py -- Inference using GPT on Track 2
│   gpt_track3.py -- Inference using GPT on Track 3
│   hf_track1.py -- Inference using HuggingFace on Track 1
│   hf_track2.py -- Inference using HuggingFace on Track 2
│   hf_track3.py -- Inference using HuggingFace on Track 3
│   explicit_querying_template.py -- Script for generating Track 2: explicit application questions
│   implicit_querying_template.py -- Script for generating Track 3: implicit application questions
│   query_gpt.py -- GPT querying API script
└───setup/
    |   requirements.txt/
└───dataset/
    │   confident_questions.csv -- csv file with NEWTON Benchmark Track 1 Questions
    |   explicit_questions.csv -- csv file with NEWTON Benchmark Track 2 Questions
    |   implicit_questions.csv -- csv file with NEWTON Benchmark Track 3 Questions
    └───dataset/ (store dataset files here)
└───utils/
    │   filter_generate.py -- utilities related to data filtering and template generation
    |   huggingface_models.py -- classes for different huggingface models
```
</details>

## :hammer: Environment Setup
<details open>
<summary>[Click to view]</summary>

We recommend setting up Anaconda to contain all necessary dependencies. To set this up, do the following:
```
$ cd PATH/TO/Newton
```

### 1. Set up the Conda Environment
Running the following command will create an Anaconda environment with the name NEWTON.
```
$ conda create --name NEWTON --file requirements.txt
```
You can activate the conda environment using:
```
conda create --name NEWTON --file requirements.txt
```

</details>

## Reproducing NEWTON Benchmark Track 2 & 3 QA Templates
<details open>
<summary>[Click to view]</summary>

```
# Generating Track 2 Questions
$ cd PATH/TO/Newton
$ python explicit_querying_template.py

# Generating Track 3 Questions
$ cd PATH/TO/Newton
$ python implicit_querying_template.py
```

</details>

## Evaluating Language Models
<details open>
<summary>[Click to view]</summary>

### 1. Set up openai credentials
Change Line 2 and 3 of ```query_gpt.py``` to your organization and api key.

### 2. Set up huggingface credentials
```
$ huggingface-cli login

```
### 3. Run inference on different benchmark tracks using different models:
```
# Inference using GPT-3.5-Turbo and GPT-4 on Track 1
$ python gpt_track1.py

# Inference using GPT-3.5-Turbo and GPT-4 on Track 2
$ python gpt_track2.py

# Inference using GPT-3.5-Turbo and GPT-4 on Track 3
$ python gpt_track3.py

# Inference using Huggingface Models on Track 1
$ python hf_track1.py

# Inference using Huggingface Models on Track 2
$ python hf_track2.py

# Inference using Huggingface Models on Track 3
$ python hf_track3.py

# Finetuning using BERT
Coming soon

```
</details>

## Reproducing NEWTON Benchmark Track 2 & 3 QA Templates
<details open>
<summary>[Click to view]</summary>

```
# Generating Track 2 Questions
$ cd PATH/TO/Newton
$ python explicit_querying_template.py

# Generating Track 3 Questions
$ cd PATH/TO/Newton
$ python implicit_querying_template.py
```

</details>

## Acknowledgements

We would like to thank Faeze Brahman, Khyathi Chandu, Christoforos Mavrogiannis, Amal Nanavati, James Park, Matt Schmittle, and all members of the Personal Robotics Lab (PRL) and Robotics and State Estimation Lab (RSELab) for fruitful discussions. Yi Ru Wang is supported by the Natural Sciences and Engineering Research Council of Canada (NSERC). This work was (partially) funded by the National Science Foundation NRI (#2132848) and CHS (#2007011), DARPA RACER (#HR0011-21-C-0171), the Office of Naval Research (#N00014-17-1-2617-P00004 and #2022-016-01 UW), and Amazon.

## Coming soon...

<details open>
<summary>[Click to view]</summary>

* Huggingface API for dataset
* Annotation interface script
* Generic pipeline for synthesizing diverse scenarios

</details>
