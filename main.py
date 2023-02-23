import os
import pickle

from evaluation import infer_labels
from data_preparation import prepare_data
from evaluation import load_transition_info
from data_preparation import load_processed_data
from transition_matrix import make_transition_info
from train_sentence_classifiers import train_sentence_classifiers


if __name__ == '__main__':
    model_name = "roberta-base"
    experiment_name = "roberta_base_ensemble"
    os.makedirs("./saved_models", exist_ok=True)
    
    # Prepare Data
    prepare_data()

    # Learn Transition Models
    if not os.path.exists("saved_models/transitions.pickle"):
        transition_info = make_transition_info(combine=False)
        with open("saved_models/transitions.pickle", "wb") as sf:
            pickle.dump(transition_info, sf)

    # Learn Sentence Classifiers
    if not os.path.exists(f"saved_models/test_predictions_{experiment_name}.pickle"):
        train_sentence_classifiers(
            model_name=model_name, combined=False, experiment_name=experiment_name, num_trials=1
        )

    # Get test predictions
    train_data = load_processed_data(path="data/train_processed.csv")
    eval_data = load_processed_data(path="data/test_processed.csv")
    transitions = load_transition_info("saved_models/transitions.pickle")
    with open(f"saved_models/test_predictions_{experiment_name}.pickle", "rb") as lmf:
        lm_prediction_scores = pickle.load(lmf)

    y_pred = dict()
    print("Predicting...")

    for document_idx in eval_data.documents:
        document = eval_data.data[eval_data.data["document"] == document_idx]

        y_pred_doc = infer_labels(
            document_idx, document["text"].tolist(), document["paragraph"].tolist(), transitions, lm_prediction_scores
        )
        y_pred[document_idx] = y_pred_doc

    with open("test_predictions.pickle", "wb") as pf:
        pickle.dump(y_pred, pf)
