import pickle
import numpy as np

from typing import List
from tqdm.auto import trange
from scipy.stats import beta
from itertools import product
from collections import Counter
from transformers import pipeline
from collections import namedtuple
from collections import defaultdict
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from transition_matrix import load_processed_data
from blocked_word_mapping import blocked_word_mapping

TransitionInfo = namedtuple(
    "TransitionInfo",
    field_names=[
        "local_transition_scores", "prior_transition_scores", "global_transition_scores", "position_model_parameters",
        "label2idx", "idx2label", "labels", "all_global_states"
    ]
)


def load_sentence_classifier(model_type: str = "roberta-base", trial: int = None):
    tokenizer = AutoTokenizer.from_pretrained(model_type, truncate=True, model_max_length=512)
    sentence_classifier = pipeline(
        task="text-classification",
        model=f"saved_models/sentence_classifier/{model_type}" + (f"_{trial}" if trial is not None else ""),
        tokenizer=tokenizer,
        return_all_scores=True,
    )

    return sentence_classifier


def load_transition_info(path: str = "saved_models/transitions.pickle"):
    with open(path, "rb") as tsf:
        return TransitionInfo(**pickle.load(tsf))


def get_position_log_score(position, params: np.ndarray):
    position = np.clip(position, 0.01, 0.99)
    return np.log(
            (1 - params[4]) * beta.pdf(position, params[0], params[1]) +
            params[4] * beta.pdf(position, params[2], params[3])
    )


def get_sentence_classifier_scores(sentence_classifier, texts, paragraph_mapping, label_set, label2idx):
    timesteps = len(set(paragraph_mapping))
    num_sentences_per_paragraph = Counter(paragraph_mapping)

    # Get label prediction scores for individual sentences
    sentence_score_matrix = np.full((timesteps, len(label_set)), fill_value=1e-9)
    sentence_scores = sentence_classifier(texts, batch_size=1)

    for t in range(len(texts)):
        sentence_scores_t = sentence_scores[t]
        paragraph_idx = paragraph_mapping[t]
        for label_score in sentence_scores_t:
            score = label_score["score"] / num_sentences_per_paragraph[paragraph_idx]
            sentence_score_matrix[paragraph_idx, label2idx[label_score["label"]]] += score

    sentence_scores = np.log(sentence_score_matrix)
    return sentence_scores


def decode_backpointers(state_scores, backpointers, idx2label):
    predicted_labels = []
    best_final_state = np.argmax(state_scores[-1])
    best_final_state, best_final_label = np.unravel_index(best_final_state, state_scores[-1].shape)
    predicted_labels.append(idx2label[best_final_label])

    last_backpointer = backpointers[-1][best_final_state, best_final_label].tolist()
    last_backpointer = tuple(last_backpointer)

    while len(predicted_labels) < len(state_scores):
        t = len(state_scores) - len(predicted_labels) - 1

        predicted_labels.append(idx2label[last_backpointer[1]])
        last_backpointer = backpointers[t][last_backpointer[0], last_backpointer[1], :]
        last_backpointer = tuple(last_backpointer.tolist())

    predicted_labels = list(reversed(predicted_labels))
    return predicted_labels


def get_allowed_labels(paragraph_sentences: List[str], relative_position: float):
    # return list(sorted(set(blocked_word_mapping.keys())))
    if all(sentence.isupper() for sentence in paragraph_sentences):
        return ["PREAMBLE", "NONE"]

    if any("was delivered by" in sentence for sentence in paragraph_sentences):
        return ["NONE"]

    if any("costs" in sentence for sentence in paragraph_sentences) and relative_position >= 0.9:
        return ["RPC"]

    if any("petition" in sentence or "appeal" in sentence for sentence in paragraph_sentences):
        if any("is dismissed" in sentence or "is allowed" in sentence for sentence in paragraph_sentences):
            return ["RPC"]

    allowed_labels = set(blocked_word_mapping.keys())
    for label, phrases in blocked_word_mapping.items():
        if any(phrase in sentence for phrase, sentence in product(phrases, paragraph_sentences)):
            allowed_labels.remove(label)

    return list(sorted(allowed_labels))


def infer_labels(doc_idx: int, sentences: List[str], paragraph_mapping: List[int], transition_info: TransitionInfo,
                 language_model_scores, use_lm_scores: bool = True, use_position_scores: bool = True,
                 use_local_transition_scores: bool = True, use_global_transition_scores: bool = True,
                 use_domain_knowledge: bool = True) -> List[str]:
    # Make global state index -> index mapping
    powers_of_2 = np.power(2, np.arange(transition_info.all_global_states.shape[1])[::-1])

    assert any([use_lm_scores, use_position_scores, use_local_transition_scores, use_global_transition_scores])
    global_transition_scores = transition_info.global_transition_scores
    local_transition_scores = transition_info.local_transition_scores
    prior_transition_scores = transition_info.prior_transition_scores

    def global_state_index(state):
        return (state * powers_of_2).sum()

    # Make paragraph -> sentence indices mapping
    paragraph2sentence_idx = defaultdict(list)
    for t, paragraph_idx in enumerate(paragraph_mapping):
        paragraph2sentence_idx[paragraph_idx].append(t)

    # Load Language Model Prediction Scores
    trials = list(sorted(language_model_scores.keys()))
    language_model_scores = np.stack([language_model_scores[k][doc_idx] for k in trials]).sum(axis=0)
    language_model_scores = np.log(language_model_scores)

    # Get position scores for all paragraphs
    num_paragraphs = len(set(paragraph_mapping))
    relative_positions = np.linspace(0.01, 0.99, num_paragraphs)
    position_scores = np.zeros((num_paragraphs, len(transition_info.labels)))

    for label in transition_info.labels:
        position_log_scores = get_position_log_score(
            relative_positions, params=transition_info.position_model_parameters[label]
        )
        position_scores[:, transition_info.label2idx[label]] = position_log_scores

    # Get the best path (Viterbi)
    state_scores = []
    backpointers = []

    for t in trange(num_paragraphs):
        # Get paragraph sentences
        paragraph_sentences = [sentences[idx] for idx in paragraph2sentence_idx[t]]

        if use_domain_knowledge:
            allowed_labels = get_allowed_labels(
                paragraph_sentences=paragraph_sentences, relative_position=t / num_paragraphs
            )
        else:
            allowed_labels = transition_info.labels

        # Initialise scores
        state_scores_t = np.full(
            shape=(len(transition_info.all_global_states), len(transition_info.labels)),
            fill_value=-100000.,
            dtype=np.float64
        )
        backpointers_t = np.zeros(
            shape=(len(transition_info.all_global_states), len(transition_info.labels), 2),
            dtype=np.int32
        )

        # Handle first paragraph separately
        if t == 0:
            for label, label_index in transition_info.label2idx.items():
                global_state_idx = 2 ** label_index + 1

                state_scores_t[global_state_idx, label_index] = (
                        (language_model_scores[t, label_index] if use_lm_scores else 0) +
                        (global_transition_scores[0, label_index] if use_global_transition_scores else 0) +
                        (position_scores[t, label_index] if use_position_scores else 0) +
                        (prior_transition_scores[label_index] if use_local_transition_scores else 0)
                )

        else:
            last_paragraph_sentences = [sentences[idx] for idx in paragraph2sentence_idx[t-1]]
            start_judgement = (
                "JUDGMENT" in last_paragraph_sentences[-1] or
                "JUDGEMENT" in last_paragraph_sentences[-1] or
                "J U D G E M E N T" in last_paragraph_sentences[-1]
            )

            has_judgement_date = any(
                "Date of Judgment" in sentence or "DATE OF JUDGMENT" in sentence
                for sentence in last_paragraph_sentences
            )
            single_judgement = last_paragraph_sentences[-1].strip() == "JUDGMENT"

            # Beam search: Only consider top scoring previous states
            prev_state_scores = state_scores[-1]
            relevant_state_indices = np.argsort(np.max(prev_state_scores, axis=1))[::-1][:100]
            relevant_state_indices = relevant_state_indices.tolist()

            for prev_global_state_index in relevant_state_indices:
                prev_global_state = transition_info.all_global_states[prev_global_state_index]

                # Iterate over all combinations of current label and previous label
                for prev_label, prev_label_idx in transition_info.label2idx.items():
                    for next_label in allowed_labels:
                        next_label_idx = transition_info.label2idx[next_label]

                        # Get global state when predicting current label
                        if prev_label == next_label:
                            next_global_state = prev_global_state
                        else:
                            next_global_state = prev_global_state.copy()
                            next_global_state[next_label_idx] = 1

                        next_global_state_idx = global_state_index(next_global_state)

                        if use_domain_knowledge and (start_judgement or single_judgement):
                            if has_judgement_date:
                                if prev_label == "PREAMBLE" and next_label != "PREAMBLE":
                                    continue

                            elif prev_label == "PREAMBLE" and next_label == "PREAMBLE":
                                continue

                        # Calculate score for current state
                        score = prev_state_scores[prev_global_state_index, prev_label_idx]
                        if use_lm_scores:
                            score += language_model_scores[t, next_label_idx]
                        if use_global_transition_scores:
                            score += global_transition_scores[prev_global_state_index, next_label_idx]
                        if use_local_transition_scores:
                            score += local_transition_scores[prev_label_idx, next_label_idx]
                        if use_position_scores:
                            score += position_scores[t, next_label_idx]

                        # Update best score if necessary
                        if score > state_scores_t[next_global_state_idx, next_label_idx]:
                            state_scores_t[next_global_state_idx, next_label_idx] = score
                            backpointers_t[next_global_state_idx, next_label_idx, 0] = prev_global_state_index
                            backpointers_t[next_global_state_idx, next_label_idx, 1] = prev_label_idx

        state_scores.append(state_scores_t)
        backpointers.append(backpointers_t)

    return decode_backpointers(
        state_scores=state_scores, backpointers=backpointers, idx2label=transition_info.idx2label
    )


if __name__ == '__main__':
    eval_data = load_processed_data(path="data/dev_processed.csv")
    # sentence_clf = load_sentence_classifier()
    transitions = load_transition_info()

    with open("saved_models/dev_predictions_ensemble_legalbert_no_context.pickle", "rb") as lmf:
        lm_prediction_scores = pickle.load(lmf)

    y_pred = []
    y_true = []

    for document_idx in eval_data.documents:
        document = eval_data.data[eval_data.data["document"] == document_idx]

        y_true_doc = eval_data.paragraph_labels[document_idx]
        y_pred_doc = infer_labels(
            document_idx, document["text"].tolist(), document["paragraph"].tolist(), transitions, lm_prediction_scores
        )
        assert len(y_true_doc) == len(y_pred_doc)

        y_true.extend(y_true_doc)
        y_pred.extend(y_pred_doc)
        print(f"Current Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0))
    print(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0.0, average="micro"))
