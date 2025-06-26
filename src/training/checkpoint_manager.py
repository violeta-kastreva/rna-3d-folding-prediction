import os
import json
import glob
from typing import Optional, Any, Union, Callable
from collections import deque

import torch

from model.utils import default, exists
from schedulers.scheduler_manager import SchedulerManager
from training.bounded_priority_queue import BoundedPriorityQueue


DataType = str


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints during training.
    Saves:
      - 'last.pth' every time save() is called.
      - 'best.pth' when the monitored metric improves.
      - Optionally keeps top-k best checkpoints in a folder, with names including epoch and metric.
    """
    def __init__(
        self,
        ckpt_dir: str,
        mode: str = "min",
        monitor: str = "val_loss",
        top_k: int = 20,
        top_k_best: Optional[int] = None,
    ):
        """
        Args:
            ckpt_dir: directory to save checkpoints.
            mode: "min" if lower metric is better, "max" if higher is better.
            monitor: the key name of the metric to monitor (string). Checkpoint dict should include this in `metrics`.
            top_k: how many best checkpoints to keep (besides 'last.pth'). If top_k=1, only best.pth is kept.
        """
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.ckpt_dir: str = ckpt_dir
        self.mode: str = mode
        self.monitor: str = monitor
        self.top_k: int = top_k
        self.top_k_best: int = default(top_k_best, top_k)

        best_queue_mode: str = "min" if mode == "max" else "max"

        self.record_file: str = os.path.join(self.ckpt_dir, "checkpoint_record.json")
        self.record: dict[str, Union[BoundedPriorityQueue[DataType], deque[DataType]]] = {}
        if os.path.exists(self.record_file):
            try:
                with open(self.record_file, "r") as f:
                    data = json.load(f)
                    self.record["last"] = deque(data["last"], maxlen=top_k)
                    self.record["best"] = BoundedPriorityQueue[DataType](
                        capacity=top_k_best,
                        mode=best_queue_mode,
                        data=data["best"],
                    )
            except Exception:
                self.record = {
                    "last": deque[DataType](maxlen=top_k),
                    "best": BoundedPriorityQueue[DataType](capacity=top_k_best, mode=best_queue_mode),
                }
        else:
            self.record = {
                "last": deque[DataType](maxlen=top_k),
                "best": BoundedPriorityQueue[DataType](capacity=top_k_best, mode=best_queue_mode),
            }

    def has_any(self) -> bool:
        return len(self.record["last"]) > 0

    def _is_better(self, metric: float, ref: float) -> bool:
        return metric < ref if self.mode == "min" else metric > ref

    def _better_function(self) -> Callable:
        return min if self.mode == "min" else max

    def _save_record(self):
        with open(self.record_file, "w") as f:
            json.dump({
                "last": list(self.record["last"]),
                "best": self.record["best"].items(),
            }, f, indent=2)

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metric: Optional[float] = None,
    ):
        # Prepare checkpoint dict
        ckpt = state

        # Remove old checkpoints if necessary
        last_queue: deque[DataType] = self.record["last"]
        if len(last_queue) >= last_queue.maxlen:
            file_to_be_removed: str = last_queue.popleft()
            self._remove_file(file_to_be_removed)

        # 1) Save last checkpoint
        last_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch}.ckpt")
        last_queue.append(last_path)
        torch.save(ckpt, last_path)

        # 2) Check for best
        if exists(metric):
            metric_value = float(metric)
            best_list: BoundedPriorityQueue[DataType] = self.record["best"]

            if self._is_better(
                    metric_value if self.mode == "min" else -metric_value,
                    self._better_function()(best_list.items(), key=lambda x: x[0], default=0)
            ):
                filename = f"best_epoch_{epoch}_{metric_value:.4f}.ckpt"
                best_path = os.path.join(self.ckpt_dir, filename)
                torch.save(ckpt, best_path)

                file_to_be_removed: Optional[str] = best_list.push(metric_value, filename)

                if file_to_be_removed is not None:
                    self._remove_file(file_to_be_removed)

        self._save_record()

    def load_last(self) -> dict[str, Any]:
        last_path = os.path.join(self.ckpt_dir, self.record["last"][-1])
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"No checkpoint at {last_path}")

        ckpt = torch.load(last_path, map_location="cpu")
        return ckpt

    def load_best(self) -> dict[str, Any]:
        best_list: BoundedPriorityQueue[DataType] = self.record["best_list"]
        filename = best_list.sorted_items(reverse=self.mode == "max")[0][1]
        best_path = os.path.join(self.ckpt_dir, filename)

        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best checkpoint at {best_path}")

        ckpt = torch.load(best_path, map_location="cpu")
        return ckpt

    def list_best(self):
        """
        Return the list of best checkpoints info: list of dicts {"path", "metric"}.
        """
        return list(self.record["best_list"].sorted_items(reverse=self.mode == "max"))

    def _remove_file(self, filename: str) -> None:
        """
        Remove a specific checkpoint file from the record and disk.
        Args:
            filename: name of the checkpoint file to remove.
        """
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
