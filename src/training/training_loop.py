import traceback
from typing import cast, Any

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import gc

from data.data_manager import DataManager
from data.typing import DataBatch, DataBatchKey
from loss.topological_loss import TopologicalLoss
from model.ribonanza_net_3d import RibonanzaNet3D
# from schedulers.scheduler_manager import SchedulerManager
from training.checkpoint_manager import CheckpointManager
from training.file_system_manager import FileSystemManager
from training.gradient_auto_clipper import GradientAutoClipper
from training.tensorboard_logger import TensorBoardLogger
from training.time_ticker import TimeTicker
from training.training_config import TrainingConfig


class TrainingLoop:
    def __init__(
            self,
            config: TrainingConfig,
            data_manager: DataManager,
            model: RibonanzaNet3D,
            loss: TopologicalLoss,
            # scheduler_manager: SchedulerManager,
            optimizer: torch.optim.Optimizer,
            file_system_manager: FileSystemManager,
            checkpoint_manager: CheckpointManager,
            tensorboard_logger: TensorBoardLogger,
            gradient_auto_clipper: GradientAutoClipper,
            # lr_scheduler: LRScheduler,
    ):
        self.config: TrainingConfig = config
        self.data_manager: DataManager = data_manager
        self.model: RibonanzaNet3D = model
        self.loss: TopologicalLoss = loss
        # self.scheduler_manager: SchedulerManager = scheduler_manager
        self.optimizer: torch.optim.Optimizer = optimizer
        self.file_system_manager: FileSystemManager = file_system_manager
        self.checkpoint_manager: CheckpointManager = checkpoint_manager
        self.tensorboard_logger: TensorBoardLogger = tensorboard_logger
        self.gradient_auto_clipper: GradientAutoClipper = gradient_auto_clipper
        # self.lr_scheduler: LRScheduler = lr_scheduler

        self.start_epoch: int = 0
        self.epoch: int = self.start_epoch
        self.time_ticker: TimeTicker = TimeTicker()

    def run(self):
        self.setup()
        self._loop()

    def _loop(self):
        try:
            self.model.to(self.config.gpu)
            for self.epoch in range(self.start_epoch, self.config.epochs):
                self._train_epoch(self.epoch)
                self._validation_epoch(self.epoch)
        except torch.OutOfMemoryError:
            print("An error occurred during training:")
            traceback.print_exc()
            print(torch.cuda.memory_summary())
            self.finish()
            gc.collect()
            torch.cuda.empty_cache()
            raise
        except Exception:
            print("An error occurred during training:")
            traceback.print_exc()
            self.finish()
            raise

    def test(self, inference_mode: bool = True):
        try:
            if inference_mode:
                self.setup()

            self._test_epoch()
        except Exception:
            print("An error occurred during testing:")
            traceback.print_exc()
            self.finish()
            gc.collect()
            torch.cuda.empty_cache()
            raise

    def setup(self) -> None:
        self.file_system_manager.ensure_created()

        if self.checkpoint_manager.has_any():
            self.load_state_dict(self.checkpoint_manager.load_last())

        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging gradient-breaking inplace operations

        self.tensorboard_logger.setup_tensorboard(self.model, device=self.config.gpu)

    def finish(self) -> None:
        """
        Finalizes the training loop, saving the final model and logging the end of training.
        """
        self.checkpoint_manager.save(self.state_dict(), epoch=self.epoch)
        self.tensorboard_logger.close()

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_state": self.loss.state_dict(),
            # "gradient_auto_clipper_state": self.gradient_auto_clipper.state_dict(),
            # "lr_scheduler_state": self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.config = state_dict["config"]
        self.start_epoch = self.epoch = state_dict["epoch"]
        self.model.load_state_dict(state_dict["model_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.loss.load_state_dict(state_dict["loss_state"])
        # self.gradient_auto_clipper.load_state_dict(state_dict["gradient_auto_clipper_state"])
        # self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state"])

    def _train_epoch(self, epoch: int):
        self.model.train()
        for i, batch in enumerate(self.data_manager.train_dataloader):
            batch = self._move_batch_to_gpu(batch)
            self.time_ticker.tick("train_batch_available")

            self.time_ticker.tick("forward_pass_start")
            output, _ = self.model(self._get_input(batch))
            self.time_ticker.tick("forward_pass_finished")

            loss, agg_separate_losses, separate_losses = self.loss(output, batch)
            self.time_ticker.tick("backward_pass_start")
            loss.backward()
            self.time_ticker.tick("backward_pass_finished")

            self.optimizer.zero_grad()

            pre_clipped_grad_norm = self.gradient_auto_clipper.get_gradient_norm()
            # self.gradient_auto_clipper.step()
            # post_clipped_grad_norm = self.gradient_auto_clipper.get_gradient_norm()

            self.time_ticker.tick("optimizer_step_start")
            self.optimizer.step()
            self.time_ticker.tick("optimizer_step_finished")
            post_clipped_grad_norm = self.gradient_auto_clipper.get_gradient_norm()

            self._train_log(
                step=epoch * len(self.data_manager.train_dataset) + i,
                epoch=epoch,
                batch_idx=i,
                batch=batch,
                output=output,
                loss=loss,
                agg_separate_losses=agg_separate_losses,
                separate_losses=separate_losses,
                pre_clipped_grad_norm= pre_clipped_grad_norm,
                post_clipped_grad_norm=post_clipped_grad_norm,
            )

            self.time_ticker.tick("last_train_batch_finished")

    def _validation_epoch(self, epoch: int):
        self.model.eval()
        forward_pass_times: list[float] = []
        losses: list[float] = []
        rmse_losses: list[float] = []
        cross_distance_losses: list[float] = []
        folding_angles_losses: list[float] = []
        probability_losses: list[float] = []
        with torch.no_grad():
            for i, batch in enumerate(self.data_manager.validation_dataloader):
                batch = self._move_batch_to_gpu(batch)

                self.time_ticker.tick("forward_pass_start")
                output, _ = self.model(self._get_input(batch))
                self.time_ticker.tick("forward_pass_finished")

                loss, agg_separate_losses, _ = self.loss(output, batch)

                forward_pass_times.append(
                    self.time_ticker.get_elapsed_secs("forward_pass_start", "forward_pass_finished")
                )

                (rmse_loss, cross_distance_loss, folding_angles_loss, probability_loss) = agg_separate_losses
                losses.append(loss.item())
                rmse_losses.append(rmse_loss.item())
                cross_distance_losses.append(cross_distance_loss.item())
                folding_angles_losses.append(folding_angles_loss.item())
                probability_losses.append(probability_loss.item())

            loss = self._validation_log(
                step=(epoch + 1) * len(self.data_manager.train_dataset),
                epoch=epoch,
                forward_pass_times=forward_pass_times,
                losses=losses,
                rmse_losses=rmse_losses,
                cross_distance_losses=cross_distance_losses,
                folding_angles_losses=folding_angles_losses,
                probability_losses=probability_losses,
            )

        self.checkpoint_manager.save(self.state_dict(), epoch=epoch, metric=loss)
        # self.lr_scheduler.step()

    def _test_epoch(self):
        print("Starting testing...")
        self.model.eval()
        forward_pass_times: list[float] = []
        losses: list[float] = []
        with torch.no_grad():
            for i, batch in enumerate(self.data_manager.test_dataloader):
                batch = self._move_batch_to_gpu(batch)

                self.time_ticker.tick("forward_pass_start")
                output, _ = self.model(self._get_input(batch))
                self.time_ticker.tick("forward_pass_finished")

                loss, *_ = self.loss(output, batch)

                forward_pass_times.append(
                    self.time_ticker.get_elapsed_secs("forward_pass_start", "forward_pass_finished")
                )
                losses.append(loss.item())

            forward_pass_times_mean = float(np.mean(forward_pass_times))
            losses_mean = float(np.mean(losses))

            print(
                f"TEST: Loss: {losses_mean}, "
                f"Forward Pass Mean Time: {forward_pass_times_mean:.2f} s, "
            )

    def _move_batch_to_gpu(self, batch: DataBatch) -> DataBatch:
        return {
            cast(DataBatchKey, key):
            value.to(device=self.config.gpu)
            if isinstance(value, torch.Tensor)
            else ([v.to(device=self.config.gpu) for v in value]
            if isinstance(value, list) and all(isinstance(v, torch.Tensor) for v in value)
            else value)
            for key, value in batch.items()
        }

    @staticmethod
    def _get_input(batch: DataBatch) -> DataBatch:
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
                "product_sequences_indices",
                "is_synthetic",
            )
        }

    def _train_log(
        self,
        step: int,
        epoch: int,
        batch_idx: int,
        batch: DataBatch,
        output: torch.Tensor,
        loss: torch.Tensor,
        agg_separate_losses: tuple[torch.Tensor, ...],
        separate_losses: tuple[torch.Tensor, ...],
        pre_clipped_grad_norm: torch.Tensor,
        post_clipped_grad_norm: torch.Tensor,
    ) -> None:

        if batch_idx % self.config.train_tensorboard_log_frequency == 0:
            tbl = self.tensorboard_logger
            (rmse_loss, cross_distance_loss, folding_angles_loss, probability_loss) = agg_separate_losses
            tbl.add_scalar("Epoch/Train", epoch, step)
            tbl.add_scalar("Loss/Train", loss, step, as_tensor=True)
            tbl.add_scalar("Loss/RMSE/Train",           rmse_loss,           step, as_tensor=True)
            tbl.add_scalar("Loss/Cross Distance/Train", cross_distance_loss, step, as_tensor=True)
            tbl.add_scalar("Loss/Folding Angles/Train", folding_angles_loss, step, as_tensor=True)
            tbl.add_scalar("Loss/Probability/Train",    probability_loss,    step, as_tensor=True)
            tbl.add_scalar("Learning Rate", self._get_lr(), step)
            tbl.add_scalar("Gradient Norm/Pre Clipped", pre_clipped_grad_norm, step, as_tensor=True)
            tbl.add_scalar("Gradient Norm/Post Clipped", post_clipped_grad_norm, step, as_tensor=True)

            self._tensorboard_log_train_times(step)

            if ((batch_idx + 1) % self.config.train_console_log_frequency == 0 or
                    batch_idx == len(self.data_manager.train_dataset) - 1):
                time_elapsed = self.time_ticker.get_elapsed_secs(
                    "train_batch_available",
                    "optimizer_step_finished",
                )

                print(
                    f"TRAIN: Epoch [{epoch + 1}/{self.config.epochs}], "
                    f"Step [{batch_idx + 1}/{len(self.data_manager.train_dataset)}], "
                    f"Loss: {loss.item()}, "
                    f"Batch Time: {time_elapsed:.2f} s, "
                    f"Overall Time: {self.time_ticker.get_time_since_start(stringify=True)}"
                )

    def _validation_log(
        self,
        step: int,
        epoch: int,
        forward_pass_times: list[float],
        losses: list[float],
        rmse_losses: list[float],
        cross_distance_losses: list[float],
        folding_angles_losses: list[float],
        probability_losses: list[float],
    ) -> float:
        forward_pass_times_mean = float(np.mean(forward_pass_times))
        losses_mean = float(np.mean(losses))
        rmse_losses_mean = float(np.mean(rmse_losses))
        cross_distance_losses_mean = float(np.mean(cross_distance_losses))
        folding_angles_losses_mean = float(np.mean(folding_angles_losses))
        probability_losses_mean = float(np.mean(probability_losses))

        tbl = self.tensorboard_logger
        tbl.add_scalar("Loss/Avg/Val",                losses_mean,                step)
        tbl.add_scalar("Loss/RMSE/Avg/Val",           rmse_losses_mean,           step)
        tbl.add_scalar("Loss/Cross Distance/Avg/Val", cross_distance_losses_mean, step)
        tbl.add_scalar("Loss/Folding Angles/Avg/Val", folding_angles_losses_mean, step)
        tbl.add_scalar("Loss/Probability/Avg/Val",    probability_losses_mean,    step)
        tbl.add_scalar("Time/Forward Pass/Avg/Val",   forward_pass_times_mean,    step)

        for i, (
            froward_pass_time,
            loss,
            rmse_loss,
            cross_distance_loss,
            folding_angles_loss,
            probability_loss,
        ) in enumerate(zip(
            forward_pass_times,
            losses,
            rmse_losses,
            cross_distance_losses,
            folding_angles_losses,
            probability_losses
        )):
            step_i = step - i
            tbl.add_scalar(f"Batch/Forward Pass Time/Val",   froward_pass_time,   step_i)
            tbl.add_scalar(f"Batch/Loss/Val",                loss,                step_i)
            tbl.add_scalar(f"Batch/RMSE Loss/Val",           rmse_loss,           step_i)
            tbl.add_scalar(f"Batch/Cross Distance Loss/Val", cross_distance_loss, step_i)
            tbl.add_scalar(f"Batch/Folding Angles Loss/Val", folding_angles_loss, step_i)
            tbl.add_scalar(f"Batch/Probability Loss/Val",    probability_loss,    step_i)

        print(
            f"VAL: Epoch [{epoch + 1}/{self.config.epochs}], "
            f"Loss: {losses_mean}, "
            f"Forward Pass Mean Time: {forward_pass_times_mean:.2f} s, "
            f"Overall Time: {self.time_ticker.get_time_since_start(stringify=True)}"
        )

        return losses_mean

    def _get_lr(self):
        """
        Returns the current learning rate of the optimizer.
        """
        return self.optimizer.param_groups[0]['lr']

    def _tensorboard_log_train_times(self, step: int):
        batch_loading_time: float = self.time_ticker.get_elapsed_secs(
            "last_train_batch_finished",
            "train_batch_available",
        ) if step != 0 else 0.0
        forward_pass_time: float = self.time_ticker.get_elapsed_secs(
            "forward_pass_start",
            "forward_pass_finished",
        )
        backward_pass_time = self.time_ticker.get_elapsed_secs(
            "backward_pass_start",
            "backward_pass_finished",
        )
        optimizer_step_time: float = self.time_ticker.get_elapsed_secs(
            "optimizer_step_start",
            "optimizer_step_finished",
        )

        self.tensorboard_logger.add_scalar("Time/Batch Loading/Train", batch_loading_time, step)
        self.tensorboard_logger.add_scalar("Time/Forward Pass/Train", forward_pass_time, step)
        self.tensorboard_logger.add_scalar("Time/Backward Pass/Train", backward_pass_time, step)
        self.tensorboard_logger.add_scalar("Time/Optimizer Step/Train", optimizer_step_time, step)
