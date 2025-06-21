import torch

from data.data_manager import DataManager
from schedulers.scheduler_manager import SchedulerManager


class TrainingLoop:
    def __init__(
            self,
    ):
        self.data_manager: DataManager = None
        self.model: torch.nn.Module = None
        self.sheduler_manager: SchedulerManager = None
        self.optimizer: torch.optim.Optimizer = None
        self.start_epoch: int = 0

    def run(self):
        ...

    def loop(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            self.sheduler_manager.step_pre_epoch(
                epoch=epoch,
                model=self.model,
                data_manager=self.data_manager,
                optimizer=self.optimizer,
            )
            self._train_epoch()
            self.sheduler_manager.step_pre_validation(
                epoch=epoch,
                model=self.model,
                data_manager=self.data_manager,
                optimizer=self.optimizer,
            )
            self._validation_epoch()
            self.sheduler_manager.step_post_validation(
                epoch=epoch,
                model=self.model,
                data_manager=self.data_manager,
                optimizer=self.optimizer,
            )
            # Save model checkpoint
            ...


    def _train_epoch(self):
        for i, batch in self.data_manager.train_dataloader:
            self.sheduler_manager.step_pre_train_batch(
                batch=batch,
                batch_idx=i,
                model=self.model,
                data_manager=self.data_manager,
                optimizer=self.optimizer,
            )
            output = self.model(batch)
            loss = self.model.compute_loss(output, batch)
            loss.backward()
            self.sheduler_manager.step_post_train_batch(
                batch=batch,
                model_output=output,
                loss=loss,
                model=self.model,
                data_manager=self.data_manager,
                optimizer=self.optimizer,
            )

    def _validation_epoch(self):
        ...

    def _test_epoch(self):...
