import torch
import pickle
import numpy as np
import torch.nn as nn

from typing import List
from tqdm.auto import tqdm
from scipy.stats import beta
from torch.optim import AdamW
from collections import defaultdict
from scipy.optimize import minimize
from data_preparation import TrainData
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from scipy.spatial.distance import euclidean
from data_preparation import load_processed_data


class GlobalTransitionMLP:
    def __init__(self, num_states: int, hidden_size: int = 256, dropout: float = 0.0, epochs=20):
        self.num_states = num_states
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs

        self.mlp = nn.Sequential(
            nn.Linear(self.num_states, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, self.num_states)
        )
        self.mlp = self.mlp.cuda()
        self.optimizer = AdamW(self.mlp.parameters())
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()
        self.log_sigmoid = nn.LogSigmoid()

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float().clamp(0.0, 1.0)

        dataloader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
        pbar = tqdm("Global Transitions Training Progress", total=self.epochs * len(dataloader))
        self.mlp.train()

        for epoch in range(self.epochs):
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                loss = self.binary_cross_entropy(self.mlp(x_batch.cuda()), y_batch.cuda())
                loss.backward()
                self.optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss.detach().cpu().item():.4f}")

        pbar.close()

    def predict_log_proba(self, x: np.ndarray):
        self.mlp.eval()

        with torch.no_grad():
            return self.log_sigmoid(self.mlp(torch.from_numpy(x).float().cuda())).cpu().numpy()


def build_global_transition_matrix(data: TrainData):
    """
    Learns a model to map states with binary indicators of already predicted labels to probability
    distributions over labels. This contains information about the global structure of judgements
    """
    # Make global states observed in training data:
    # Binary vectors whose components indicate which labels have been predicted before the current paragraph
    x_train = []
    y_train = []

    for document_idx in data.documents:
        running_state = [0 for _ in data.label_set]
        document_labels = data.paragraph_labels[document_idx]
        for i, current_label in enumerate(document_labels):
            current_y = [0 for _ in data.label_set]
            for label in data.label_set:
                if label in document_labels[i:]:
                    current_y[data.label2idx[label]] = 1

            x_train.append(running_state.copy())
            y_train.append(current_y)

            running_state[data.label2idx[current_label]] = 1

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Train a transition model that maps global states to distributions over labels
    # (which next label is likely given that we have already seen the provided labels)
    # This is a joint distribution of all possible binary global state vectors and labels
    # However, due to sparsity we model the conditional distributions by a MLP

    global_transition_model = GlobalTransitionMLP(num_states=len(data.label_set))
    global_transition_model.fit(x_train, y_train)

    # Get predictions for all possible global states
    def recursive_make_global_states(n: int) -> List[List[int]]:
        if n == 0:
            return [[]]

        global_states = []
        for global_state in recursive_make_global_states(n - 1):
            global_states.append(global_state + [0])
            global_states.append(global_state + [1])

        return global_states

    all_global_states = np.array(recursive_make_global_states(len(data.label_set)))
    global_transition_matrix = global_transition_model.predict_log_proba(all_global_states)
    return all_global_states, global_transition_matrix


def build_local_transition_matrix(data: TrainData):
    """Learns transition probabilities of labels"""
    # Learn a prior (probabilities for first paragraph) and transition probabilities
    # Also apply add-1 smoothing to avoid 0 probabilities
    num_labels = len(data.label_set)
    local_transition_scores = np.full((num_labels, num_labels), fill_value=-1e8)
    prior_transition_scores = np.full(num_labels, fill_value=0.)

    # Count the number of transitions for all pairs of consecutive labels
    for document_idx in data.documents:
        prior_transition_scores[data.label2idx[data.paragraph_labels[document_idx][0]]] += 1.
        previous_label = data.paragraph_labels[document_idx][0]

        for label in data.paragraph_labels[document_idx][1:]:
            previous_label_index = data.label2idx[previous_label]
            current_label_index = data.label2idx[label]

            local_transition_scores[previous_label_index, current_label_index] = 0.
            previous_label = label

    # Normalise transition log probabilities
    prior_transition_scores = np.log(prior_transition_scores) - np.log(prior_transition_scores.sum())
    # local_transition_scores = (np.log(local_transition_scores).T - np.log(local_transition_scores.sum(axis=1))).T

    return prior_transition_scores, local_transition_scores


def build_position_model(data: TrainData):
    """Learns a distribution over labels for relative positions of paragraphs in the judgements"""
    # Build mapping from labels to relative positions where they are found
    labels2timesteps = defaultdict(list)

    for document_idx in data.documents:
        labels = data.paragraph_labels[document_idx]
        timesteps = np.linspace(0.0, 1.0, len(labels)).tolist()

        for label, t in zip(labels, timesteps):
            labels2timesteps[label].append(t)

    # Make loss function for fitting mixture of gamma distributions
    def get_loss(label_name: str):
        bins = np.linspace(0.0, 1.0, 102)
        bins[1] = 1e-8
        bins[-1] = 1 - 1e-8
        hist, _ = np.histogram(labels2timesteps[label_name], bins=bins, density=False)
        hist = np.cumsum(hist, axis=0) / np.sum(hist)

        def loss(x):
            beta_hist_1 = beta.cdf(bins[1:], x[0], x[1])
            beta_hist_2 = beta.cdf(bins[1:], x[2], x[3])
            beta_hist = (1 - x[4]) * beta_hist_1 + x[4] * beta_hist_2
            dist = euclidean(hist, beta_hist)
            return dist

        return loss

    gamma_parameters = dict()
    for label in data.label_set:
        current_loss = get_loss(label_name=label)
        optimization = minimize(
            current_loss, np.array([1, 1, 1, 1, 0.1]), method="L-BFGS-B",
            bounds=[(0.1, None), (0.1, None), (0.1, None), (0.1, None), (0, 1)]
        )
        gamma_parameters[label] = optimization.x

    return gamma_parameters


def make_transition_info(combine: bool = False):
    print("Load Data")
    path_to_data = "data/train_processed.csv" if not combine else "data/train_dev_combined.csv"
    train_data = load_processed_data(path=path_to_data)
    print("Build Global Transition Scores")
    all_global_states, global_transition_scores = build_global_transition_matrix(data=train_data)
    print("Build Local Transition Scores")
    prior_transition_scores, local_transition_scores = build_local_transition_matrix(data=train_data)
    print("Build Position Scores")
    position_model_parameters = build_position_model(data=train_data)

    return {
        "local_transition_scores": local_transition_scores,
        "prior_transition_scores": prior_transition_scores,
        "global_transition_scores": global_transition_scores,
        "position_model_parameters": position_model_parameters,
        "label2idx": train_data.label2idx,
        "idx2label": train_data.idx2label,
        "labels": train_data.label_set,
        "all_global_states": all_global_states
    }


if __name__ == '__main__':
    transition_info = make_transition_info()
    with open("saved_models/transitions.pickle", "wb") as sf:
        pickle.dump(transition_info, sf)

    print("Done")
