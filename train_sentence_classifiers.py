import torch
import pickle
import numpy as np

from tqdm.auto import tqdm
from datasets import Dataset
from collections import Counter
from transformers import Trainer
from transformers import pipeline
from transformers import AutoTokenizer
from data_preparation import TrainData
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transition_matrix import load_processed_data
from transformers import AutoModelForSequenceClassification


def train_sentence_classifier(model_name: str, experiment_name: str, train_data: TrainData, dev_data: TrainData,
                              trial: int = 0):
    # Make Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_data.label_set),
        id2label=train_data.idx2label,
        label2id=train_data.label2idx
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Make Train Data
    train_sentences = train_data.data["text"].tolist()
    train_labels = [train_data.label2idx[label] for label in train_data.data["label"].tolist()]
    train_dataset = Dataset.from_dict(
        {"sentence": train_sentences, "label": train_labels}
    )
    print(train_dataset)

    train_dataset = train_dataset.map(
        lambda examples: tokenizer(examples["sentence"], truncation=True),
        batched=True
    )

    # Make Dev data
    dev_sentences = dev_data.data["text"].tolist()
    dev_labels = [train_data.label2idx[label] for label in dev_data.data["label"].tolist()]
    dev_dataset = Dataset.from_dict(
        {"sentence": dev_sentences, "label": dev_labels}
    )

    dev_dataset = dev_dataset.map(
        lambda examples: tokenizer(examples["sentence"], truncation=True),
        batched=True
    )

    # Make Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}_{trial}",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.0,
        save_strategy="epoch",
        optim="adamw_torch",
        evaluation_strategy="epoch",
        warmup_steps=1000,
        load_best_model_at_end=True,
        seed=trial
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(f"saved_models/sentence_classifier_{experiment_name}/{model_name}_{trial}/")

    # Clean Up
    del model
    torch.cuda.empty_cache()


def get_trial_scores(model_name: str, experiment_name: str, trial: int, eval_data, train_data):
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncate=True, model_max_length=512)
    sentence_classifier = pipeline(
        task="text-classification",
        model=f"saved_models/sentence_classifier_{experiment_name}/{model_name}_{trial}",
        tokenizer=tokenizer,
        return_all_scores=True,
    )

    trial_pred_scores = dict()

    for doc_idx in tqdm(list(sorted(eval_data.documents))):
        document = eval_data.data[eval_data.data["document"] == doc_idx]

        paragraph_mapping = document["paragraph"].tolist()
        timesteps = len(set(paragraph_mapping))
        num_sentences_per_paragraph = Counter(paragraph_mapping)

        # Get label prediction scores for individual sentences
        texts = document["text"].tolist()
        sentence_score_matrix = np.full((timesteps, len(train_data.label_set)), fill_value=0., dtype=np.float64)
        sentence_scores = sentence_classifier(texts, batch_size=16)

        for t in range(len(texts)):
            sentence_scores_t = sentence_scores[t]
            paragraph_idx = paragraph_mapping[t]
            for label_score in sentence_scores_t:
                score = label_score["score"] / num_sentences_per_paragraph[paragraph_idx]
                sentence_score_matrix[paragraph_idx, train_data.label2idx[label_score["label"]]] += score

        trial_pred_scores[doc_idx] = sentence_score_matrix.copy()

    del sentence_classifier
    torch.cuda.empty_cache()

    return trial_pred_scores


def train_sentence_classifiers(model_name: str, experiment_name: str, combined: bool = False, num_trials: int = 5):
    training_data = load_processed_data("data/train_processed.csv")
    development_data = load_processed_data("data/dev_processed.csv")
    combined_data = load_processed_data("data/train_dev_combined.csv")
    test_data = load_processed_data("data/test_processed.csv")
    train_data = training_data if not combined else combined_data

    test_prediction_scores = dict()
    dev_prediction_scores = dict()

    experiment_name = experiment_name + ("_combined" if combined else "")

    for k in range(1, num_trials + 1):
        train_sentence_classifier(
            model_name=model_name, experiment_name=experiment_name, train_data=train_data, dev_data=development_data,
            trial=k
        )

    for k in range(1, num_trials + 1):
        test_prediction_scores[k] = get_trial_scores(
            trial=k, eval_data=test_data, train_data=train_data, model_name=model_name, experiment_name=experiment_name
        )

        if not combined:
            dev_prediction_scores[k] = get_trial_scores(
                trial=k, eval_data=development_data, train_data=train_data, model_name=model_name,
                experiment_name=experiment_name
            )

    with open(f"saved_models/test_predictions_{experiment_name}.pickle", "wb") as psf:
        pickle.dump(test_prediction_scores, psf)

    if not combined:
        with open(f"saved_models/dev_predictions_{experiment_name}.pickle", "wb") as psf:
            pickle.dump(dev_prediction_scores, psf)


if __name__ == '__main__':
    train_sentence_classifiers(
        model_name="zlucia/legalbert", experiment_name="ensemble_legalbert_no_context", combined=False, num_trials=5
    )
