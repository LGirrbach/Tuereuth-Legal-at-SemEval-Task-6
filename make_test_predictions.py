import json
import pickle


if __name__ == '__main__':
    with open("test_predictions.pickle", "rb") as pf:
        test_predictions = pickle.load(pf)
    
    with open("data/test.json") as data_file:
        data = json.load(data_file)

    processed_data = []
    for document_id, document in enumerate(data):
        document = document["annotations"][0]["result"]
        predicted_labels = test_predictions[document_id]
        assert len(document) == len(predicted_labels)

        for paragraph, label in zip(document, predicted_labels):
            paragraph = paragraph["value"]
            paragraph["labels"][0] = label
    
    with open("RR_TEST_DATA_FS.json", "w") as tsf:
        json.dump(data, tsf)
