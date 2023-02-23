import os
import torch
import pickle
import numpy as np

from tqdm.auto import tqdm
from collections import Counter
from transformers import pipeline
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from transition_matrix import load_processed_data


def get_trial_scores(trial: int, eval_data):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncate=True, model_max_length=512)
    sentence_classifier = pipeline(
        task="text-classification",
        model=f"saved_models/sentence_classifier/roberta-base_{trial}",
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
        sentence_score_matrix = np.full((timesteps, len(eval_data.label_set)), fill_value=0., dtype=np.float64)
        sentence_scores = sentence_classifier(texts, batch_size=16)

        for t in range(len(texts)):
            sentence_scores_t = sentence_scores[t]
            paragraph_idx = paragraph_mapping[t]
            for label_score in sentence_scores_t:
                score = label_score["score"] / num_sentences_per_paragraph[paragraph_idx]
                sentence_score_matrix[paragraph_idx, eval_data.label2idx[label_score["label"]]] += score

        trial_pred_scores[doc_idx] = sentence_score_matrix.copy()

    del sentence_classifier
    torch.cuda.empty_cache()

    return trial_pred_scores


if __name__ == '__main__':
    dev_data = load_processed_data("data/dev_processed.csv")
    sorted_documents = list(sorted(dev_data.documents))

    prediction_scores = dict()

    if os.path.exists("saved_models/dev_ensemble_predictions.pickle"):
        with open("saved_models/dev_ensemble_predictions.pickle", "rb") as psf:
            prediction_scores = pickle.load(psf)

    else:
        prediction_scores = {k: get_trial_scores(trial=k, eval_data=dev_data) for k in range(1, 6)}
        with open("saved_models/dev_ensemble_predictions.pickle", "wb") as psf:
            pickle.dump(prediction_scores, psf)

    y_true = []
    for document_idx in sorted_documents:
        y_true.extend(dev_data.paragraph_labels[document_idx])

    for model, trial_prediction_scores in prediction_scores.items():
        print(model)
        print(trial_prediction_scores[0][:2].round(2))
        trial_prediction_scores = np.concatenate(
            [trial_prediction_scores[document_idx] for document_idx in sorted_documents], axis=0
        )
        y_pred = np.argmax(trial_prediction_scores, axis=1).tolist()
        y_pred = [dev_data.idx2label[label] for label in y_pred]
        print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0))
        print("\n\n")

    y_pred_ensemble = np.argmax(
        np.stack(
            [
                np.concatenate([trial_prediction_scores[document_idx] for document_idx in sorted_documents], axis=0)
                for trial_prediction_scores in prediction_scores.values()
            ]
        ).sum(axis=0),
        axis=1
    )
    y_pred_ensemble = y_pred_ensemble.tolist()
    y_pred_ensemble = [dev_data.idx2label[label] for label in y_pred_ensemble]
    print("Ensemble:")
    print(classification_report(y_true=y_true, y_pred=y_pred_ensemble, zero_division=0.0))
    print("\n\n")
