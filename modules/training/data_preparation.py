from datasets import load_dataset, DatasetDict
import glob

def load_training_data():

    # Get a list of all JSON files recursively
    json_files = glob.glob("data/training/**/*.json", recursive=True)

    # Load all JSON files into a single dataset
    dataset = load_dataset("json", data_files=json_files)

    # Combine into a single dataset and split
    full_dataset = dataset["train"]  # all data merged
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)

    final_dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    print(final_dataset)

    return final_dataset