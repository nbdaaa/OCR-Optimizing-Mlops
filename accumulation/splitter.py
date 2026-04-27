from datasets import Dataset, DatasetDict
from common.storage import push_dataset
from common.logging import get_logger

log = get_logger(__name__)


def split_and_push(dataset: Dataset) -> DatasetDict:
    # first split off 20% for val+test
    split_1    = dataset.train_test_split(test_size=0.2, seed=42)
    # split that 20% evenly into val and test (10% each of total)
    split_2    = split_1["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict({
        "train": split_1["train"],
        "val":   split_2["train"],
        "test":  split_2["test"],
    })

    log.info(
        f"Split sizes — train: {len(dataset_dict['train'])} | "
        f"val: {len(dataset_dict['val'])} | "
        f"test: {len(dataset_dict['test'])}"
    )

    for split_name, split_data in dataset_dict.items():
        push_dataset(split_data, config_name=split_name)

    return dataset_dict