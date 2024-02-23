# qian-2021-lifelong
Implementation of "Lifelong Learning of Hate Speech Classification on Social Media" by Qian et al. for research purposes.
Main code obtained from [aclanthology](https://aclanthology.org/2021.naacl-main.183/).

# Specifications
python version: 3.5.2 \
Pytorch version: 1.4.0 \
numpy version: 1.18.2 \
sklearn verison: 0.22.2.post1 \
tokenizers version: 0.5.2 \
transformers version: 2.5.1 \
nltk version: 3.4.5 \
spacy version: 2.2.4 \
gensim version: 3.8.1 \

## other required packages:

preprocessor: https://pypi.org/project/tweet-preprocessor/

- experiments were run on the NVIDIA GeForce GTX 1080 Ti GPUs.

# How to Run?

python -m src.main --train_path TRAIN_PATH --dev_path DEV_PATH --test_path TEST_PATH --cand_limit 1000 --keep_all_node --fix_per_task_mem_size
