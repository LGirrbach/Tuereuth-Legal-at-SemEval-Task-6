# TüReuth Legal at SemEval Task 6 Code

This repository contains the code to replicate experiments for our participation at [SemEval Task 6: LegalAI](https://sites.google.com/view/legaleval/home).
Note that we only participate in subtask (A), predicting Rhetorical Roles.

## Usage
  1. Create Folder `data`
  2. Place 3 `json` files called `train.json`, `dev.json`, and `test.json` in folder `data`. These files are provided by the shared task organisers, but have to be renamed accordingly
  3. Run `python main.py`. This will train and save all models and output test predictions in file `test_predictions.pickle`.
  4. Run `python make_test_predictions.py`. This will create a file called `RR_TEST_DATA_FS.json` which contains test set predictions in the right format for submission to the shared task.

Note that training MLPs and fine-tuning LMs requires GPU and internet access to download models from huggingface model hub.

## Requirements
  * `torch` (with GPU support)
  * `transformers`
  * `datasets`
  * `nltk`
  * `numpy`
  * `scipy`
  * `pandas`
  * `tqdm`
