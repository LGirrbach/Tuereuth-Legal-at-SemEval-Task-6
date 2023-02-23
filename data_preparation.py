import json
import nltk
import pandas as pd

from typing import List
from collections import namedtuple

TrainData = namedtuple(
    "TrainData",
    field_names=["data", "documents", "label_set", "paragraph_labels", "paragraph_texts", "label2idx", "idx2label"]
)


def process_dataset(path_to_dataset: str) -> pd.DataFrame:
    with open(path_to_dataset) as data_file:
        data = json.load(data_file)

    processed_data = []
    for document_id, document in enumerate(data):
        document = document["annotations"][0]["result"]

        paragraphs, paragraph_labels = [], []
        for paragraph in document:
            paragraph = paragraph["value"]
            paragraphs.append(paragraph["text"])
            paragraph_labels.append(paragraph["labels"][0])

        sentence_counter = 0
        for paragraph_id, (paragraph, paragraph_label) in enumerate(zip(paragraphs, paragraph_labels)):
            for sentence in nltk.sent_tokenize(paragraph):
                processed_data.append(
                    {
                        "document": document_id,
                        "paragraph": paragraph_id,
                        "sentence": sentence_counter,
                        "text": sentence.strip(),
                        "label": paragraph_label
                    }
                )
                sentence_counter += 1

    processed_data = pd.DataFrame.from_records(processed_data)
    return processed_data


def get_paragraph_labels(data: pd.DataFrame) -> List[str]:
    data = data[["paragraph", "label"]]
    paragraph_labels = data.groupby(["paragraph"])["label"].first().tolist()
    return paragraph_labels


def load_processed_data(path: str = "data/train_processed.csv") -> TrainData:
    data = pd.read_csv(path, index_col=0)
    documents = list(sorted(set(data["document"].tolist())))
    label_set = list(sorted(set(data["label"].tolist())))
    label2idx = {label: idx for idx, label in enumerate(label_set)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    paragraph_labels = {
        document_idx: get_paragraph_labels(data[data["document"] == document_idx])
        for document_idx in data["document"].tolist()
    }

    paragraph_texts = dict()
    for document_idx in documents:
        document_data = data[data["document"] == document_idx]
        sentences = document_data["text"].tolist()
        paragraph_mapping = document_data["paragraph"].tolist()

        document_paragraphs = [[] for _ in set(paragraph_mapping)]
        for t, sentence in enumerate(sentences):
            document_paragraphs[paragraph_mapping[t]].append(sentence)
        document_paragraphs = [" ".join(paragraph) for paragraph in document_paragraphs]
        paragraph_texts[document_idx] = document_paragraphs

    return TrainData(
        data=data, documents=documents, label_set=label_set, label2idx=label2idx, idx2label=idx2label,
        paragraph_labels=paragraph_labels, paragraph_texts=paragraph_texts
    )


def prepare_data():
    print("Processing Train Data")
    processed_train_data = process_dataset("data/train.json")
    print("Processing Dev Data")
    processed_dev_data = process_dataset("data/dev.json")

    # Save train and dev datasets
    processed_train_data.to_csv("data/train_processed.csv")
    processed_dev_data.to_csv("data/dev_processed.csv")

    # Combine datasets
    dev_document_id_offset = processed_train_data["document"].max() + 1
    processed_dev_data["document"] = processed_dev_data["document"] + dev_document_id_offset
    combined_train_dev_data = pd.concat([processed_train_data, processed_dev_data], axis=0)
    combined_train_dev_data.to_csv("data/train_dev_combined.csv")

    # Process test data
    print("Processing Test Data")
    processed_test_data = process_dataset("data/test.json")
    processed_test_data.to_csv("data/test_processed.csv")


if __name__ == '__main__':
    prepare_data()
