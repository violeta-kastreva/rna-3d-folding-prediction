import os
import json
import glob
from typing import Optional, Any, Union, Callable
from collections import deque

import torch

from model.utils import default
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


        # Record of best checkpoints: list of dicts with keys: 'path', 'metric'
        # We'll maintain a JSON file to persist across runs.
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
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_manager: SchedulerManager,
        metrics: dict[str, float],
        **kwargs,
    ):
        # Prepare checkpoint dict
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_manager_state": scheduler_manager.state_dict(),
            **kwargs,  # allow passing additional state
        }

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
        if self.monitor in metrics:
            metric_value = float(metrics[self.monitor])
            best_list: BoundedPriorityQueue[DataType] = self.record["best"]

            if self._is_better(
                    metric_value if self.mode == "min" else -metric_value,
                    self._better_function()(best_list.items(), key=lambda x: x[0], default=0)
            ):
                filename = f"best_epoch_{epoch}_{self.monitor}_{metric_value:.4f}.ckpt"
                best_path = os.path.join(self.ckpt_dir, filename)
                torch.save(ckpt, best_path)

                file_to_be_removed: Optional[str] = best_list.push(metric_value, filename)

                if file_to_be_removed is not None:
                    self._remove_file(file_to_be_removed)

        self._save_record()

    def load_last(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim._LRScheduler = None):
        """
        Load the most recent 'last.pth' checkpoint into model (and optimizer/scheduler if provided).
        Returns:
            epoch (int), metrics (dict)
        """
        last_path = os.path.join(self.ckpt_dir, "last.pth")
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"No checkpoint at {last_path}")
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt.get("epoch", None), ckpt.get("metrics", {})

    def load_best(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim._LRScheduler = None, idx: int = 0):
        """
        Load one of the top-k best checkpoints.
        Args:
            idx: index in [0..top_k-1]; 0 is best, 1 second best, etc., sorted by monitor metric.
        Returns:
            epoch (int), metrics (dict)
        """
        best_list = self.record.get("best_list", [])
        if not best_list:
            raise FileNotFoundError("No best checkpoints recorded.")
        if idx < 0 or idx >= len(best_list):
            raise IndexError(f"idx should be in [0, {len(best_list)-1}]")
        filename = best_list[idx]["path"]
        best_path = os.path.join(self.ckpt_dir, filename)
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Checkpoint file not found: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt.get("epoch", None), ckpt.get("metrics", {})

    def list_best(self):
        """
        Return the list of best checkpoints info: list of dicts {"path", "metric"}.
        """
        return list(self.record.get("best_list", []))

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


# === Example usage in a training loop ===
#
# manager = CheckpointManager("checkpoints", mode="min", monitor="val_loss", top_k=3)
# for epoch in range(start_epoch, num_epochs):
#     train_one_epoch(...)
#     val_loss = evaluate(...)
#     metrics = {"val_loss": val_loss}
#     manager.save(epoch, model, optimizer, scheduler, metrics)
#
# # To resume:
# epoch, metrics = manager.load_last(model, optimizer, scheduler)
#
# # To load the best:
# epoch_best, metrics_best = manager.load_best(model, optimizer, scheduler, idx=0)