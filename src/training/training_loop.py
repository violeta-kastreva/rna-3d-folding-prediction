from typing import cast

import torch

from data.data_manager import DataManager
from data.typing import DataBatch, DataBatchKey
from loss.topological_loss import TopologicalLoss
from model.ribonanza_net_3d import RibonanzaNet3D
from schedulers.scheduler_manager import SchedulerManager
from training.checkpoint_manager import CheckpointManager
from training.file_system_manager import FileSystemManager
from training.memory_stack import MemoryStack
from training.training_config import TrainingConfig


class TrainingLoop:
    def __init__(
            self,
            config: TrainingConfig,
            data_manager: DataManager,
            model: RibonanzaNet3D,
            loss: TopologicalLoss,
            scheduler_manager: SchedulerManager,
            optimizer: torch.optim.Optimizer,
            file_system_manager: FileSystemManager,
            checkpoint_manager: CheckpointManager,
    ):
        self.config: TrainingConfig = config
        self.data_manager: DataManager = data_manager
        self.model: RibonanzaNet3D = model
        self.loss: TopologicalLoss = loss
        self.scheduler_manager: SchedulerManager = scheduler_manager
        self.optimizer: torch.optim.Optimizer = optimizer
        self.file_system_manager: FileSystemManager = file_system_manager
        self.checkpoint_manager: CheckpointManager = checkpoint_manager

        self.start_epoch: int = 0
        self.memory: MemoryStack = MemoryStack()

        self.memory["config"] = config
        self.memory["data_manager"] = data_manager
        self.memory["model"] = model
        self.memory["loss"] = loss
        self.memory["scheduler_manager"] = scheduler_manager
        self.memory["optimizer"] = optimizer
        self.memory["file_system_manager"] = file_system_manager
        self.memory["checkpoint_manager"] = checkpoint_manager
        self.memory["start_epoch"] = self.start_epoch

    def run(self):
        self.scheduler_manager.step_on_setup(**self.memory.to_dict())
        self.scheduler_manager.step_pre_training(**self.memory.to_dict())

        self.loop()

        self.scheduler_manager.step_post_training(**self.memory.to_dict())

    def loop(self):
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self.scheduler_manager.step_pre_epoch(**self.memory.to_dict())

                self._train_epoch()
                self._validation_epoch()

                self.scheduler_manager.step_post_epoch(**self.memory.to_dict())
        except Exception as e:
            self.scheduler_manager.step_on_train_error(**self.memory.to_dict(), error=e)

    def test(self, inference_mode: bool = True):
        try:
            if inference_mode:
                self.scheduler_manager.step_on_setup(**self.memory.to_dict())

            self.scheduler_manager.step_pre_test(**self.memory.to_dict())

            self._test_epoch()

            self.scheduler_manager.step_post_test(**self.memory.to_dict())
        except Exception as e:
            self.scheduler_manager.step_on_test_error(**self.memory.to_dict(), error=e)

    def _train_epoch(self):
        self.model.train()
        for i, batch in self.data_manager.train_dataloader:
            batch = self.move_batch_to_gpu(batch)
            self.memory["batch_idx"] = i
            self.memory["batch"] = batch

            self.scheduler_manager.step_pre_train_batch(**self.memory.to_dict())

            output, _ = self.model(self.get_input(batch))
            self.memory["output"] = output

            loss = self.model.compute_loss(output, batch)
            self.memory["loss"] = loss
            loss.backward()

            self.scheduler_manager.step_post_train_batch(**self.memory.to_dict())

    def _validation_epoch(self):
        self.model.eval()
        self.scheduler_manager.step_pre_validation(**self.memory.to_dict())

        for i, batch in self.data_manager.validation_dataloader:
            batch = self.move_batch_to_gpu(batch)
            self.memory["batch_idx"] = i
            self.memory["batch"] = batch

            self.scheduler_manager.step_pre_validation_batch(**self.memory.to_dict())

            output, _ = self.model(self.get_input(batch))
            self.memory["output"] = output

            loss = self.model.compute_loss(output, batch)
            self.memory["loss"] = loss

            self.scheduler_manager.step_post_validation_batch(**self.memory.to_dict())

        self.scheduler_manager.step_post_validation(**self.memory.to_dict())

    def _test_epoch(self):
        self.model.eval()
        for i, batch in self.data_manager.test_dataloader:
            batch = self.move_batch_to_gpu(batch)
            self.memory["batch_idx"] = i
            self.memory["batch"] = batch

            self.scheduler_manager.step_pre_test_batch(**self.memory.to_dict())

            output, _ = self.model(self.get_input(batch))
            self.memory["output"] = output

            loss = self.model.compute_loss(output, batch)
            self.memory["loss"] = loss

            self.scheduler_manager.step_post_test_batch(**self.memory.to_dict())

    def move_batch_to_gpu(self, batch: DataBatch) -> DataBatch:
        return {
            cast(DataBatchKey, key):
            value.to(device=self.config.gpu)
            if isinstance(value, torch.Tensor)
            else ([v.to(device=self.config.gpu) for v in value]
            if isinstance(value, list) and all(isinstance(v, torch.Tensor) for v in value)
            else value)
            for key, value in batch.items()
        }

    def get_input(self, batch: DataBatch) -> DataBatch:
        return {
            cast(DataBatchKey, key): batch[cast(DataBatchKey, key)]
            for key in (
                "target_id",
                "sequence",
                "sequence_mask",
                "has_msa",
                "msa",
                "msa_profiles",
                "num_product_sequences",
                "product_sequences",
                "is_synthetic",
            )
        }
