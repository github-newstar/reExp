import torch
import torch.distributed as dist
from collections import OrderedDict


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._keys = tuple(keys)
        self._data = OrderedDict(
            (k, {"total": 0.0, "counts": 0.0, "average": 0.0}) for k in self._keys
        )
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for key in self._keys:
            self._data[key]["total"] = 0.0
            self._data[key]["counts"] = 0.0
            self._data[key]["average"] = 0.0

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        if key not in self._data:
            raise KeyError(f"Unknown metric key: {key}")
        total = self._data[key]["total"] + float(value) * float(n)
        counts = self._data[key]["counts"] + float(n)
        self._data[key]["total"] = total
        self._data[key]["counts"] = counts
        self._data[key]["average"] = total / counts if counts > 0.0 else 0.0

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data[key]["average"]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return {k: self._data[k]["average"] for k in self._keys}

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._keys

    def sync_between_processes(self, device="cpu"):
        """
        Synchronize metric totals/counts across all distributed ranks.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        totals = torch.tensor(
            [self._data[k]["total"] for k in self._keys],
            dtype=torch.float64,
            device=device,
        )
        counts = torch.tensor(
            [self._data[k]["counts"] for k in self._keys],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        totals_cpu = totals.cpu().tolist()
        counts_cpu = counts.cpu().tolist()
        for idx, key in enumerate(self._keys):
            total = float(totals_cpu[idx])
            count = float(counts_cpu[idx])
            self._data[key]["total"] = total
            self._data[key]["counts"] = count
            self._data[key]["average"] = total / count if count > 0.0 else 0.0
