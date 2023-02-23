import pickle
import pandas as pd

from copy import deepcopy

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

from evaluation import infer_labels
from evaluation import load_processed_data
from evaluation import load_transition_info


def evaluate_setup(use_lm_scores: bool, use_position_scores: bool, use_local_transition_scores: bool,
                   use_global_transition_scores: bool, use_domain_knowledge: bool):
    eval_data = load_processed_data(path="data/dev_processed.csv")
    transitions = load_transition_info()

    with open("saved_models/dev_predictions_ensemble_legalbert_no_context.pickle", "rb") as lmf:
        lm_prediction_scores = pickle.load(lmf)

    y_pred = []
    y_true = []

    for document_idx in eval_data.documents:
        document = eval_data.data[eval_data.data["document"] == document_idx]

        y_true_doc = eval_data.paragraph_labels[document_idx]
        y_pred_doc = infer_labels(
            document_idx, document["text"].tolist(), document["paragraph"].tolist(), transitions, lm_prediction_scores,
            use_lm_scores=use_lm_scores, use_position_scores=use_position_scores,
            use_local_transition_scores=use_local_transition_scores,
            use_global_transition_scores=use_global_transition_scores,
            use_domain_knowledge=use_domain_knowledge
        )

        assert len(y_true_doc) == len(y_pred_doc)

        y_true.extend(y_true_doc)
        y_pred.extend(y_pred_doc)

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0.0, average="micro")
    return {"accuracy": accuracy, "f1": f1}


if __name__ == '__main__':
    score_names = [
        "use_lm_scores", "use_position_scores", "use_local_transition_scores", "use_global_transition_scores",
        "use_domain_knowledge"
    ]
    ablation_grid = {score_name: [True, False] for score_name in score_names}

    results = []
    for ablation_settings in ParameterGrid(ablation_grid):
        try:
            metrics = evaluate_setup(**ablation_settings)
        except AssertionError:
            continue

        result = deepcopy(ablation_settings)
        result.update(metrics)
        results.append(result)

    results = pd.DataFrame.from_records(results)
    results.to_csv("ablation_results_legalbert.csv")
