import numpy as np
import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    if len(dataset_items) == 0:
        return {}

    result_batch = {}
    batch_keys = dataset_items[0].keys()
    for key in batch_keys:
        values = [item[key] for item in dataset_items]
        first_value = values[0]

        if torch.is_tensor(first_value):
            result_batch[key] = torch.stack(values)
        elif isinstance(first_value, np.ndarray):
            result_batch[key] = torch.from_numpy(np.stack(values))
        elif isinstance(first_value, (int, float, bool)):
            result_batch[key] = torch.tensor(values)
        else:
            result_batch[key] = values

    return result_batch
