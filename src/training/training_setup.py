from dataclasses import asdict
from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from typing_extensions import Self

from data.data_manager import DataManager
from loss.topological_loss import TopologicalLoss
from model.ribonanza_net_3d import RibonanzaNet3D
from optimizer import Ranger21
# from schedulers.scheduler_manager import SchedulerManager
# from schedulers.scheduler_manager_builder import SchedulerManagerBuilder
from training.checkpoint_manager import CheckpointManager
from training.file_system_manager import FileSystemManager
from training.gradient_auto_clipper import GradientAutoClipper
from training.tensorboard_logger import TensorBoardLogger
from training.training_config import TrainingConfig
from training.training_loop import TrainingLoop


class TrainingSetup:
    """ TrainingLoop builder class."""
    def __init__(self, training_config: TrainingConfig):
        self.training_config: TrainingConfig = training_config
        self.data_manager: Optional[DataManager] = None  # Placeholder for DataManager type
        self.model: Optional[RibonanzaNet3D] = None
        self.optimizer: Optional[Ranger21] = None
        # self.scheduler_manager: Optional[SchedulerManager] = None
        self.loss: Optional[TopologicalLoss] = None
        self.file_system_manager: Optional[FileSystemManager] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.tensorboard_logger: Optional[TensorBoardLogger] = None
        self.gradient_auto_clipper: Optional[GradientAutoClipper] = None
        self.lr_scheduler: Optional[LRScheduler] = None

    def create_data_manager(self) -> Self:
        self.data_manager = DataManager(self.training_config.data_config, device=self.training_config.cpu)
        return self

    def create_model(self) -> Self:
        self.model = RibonanzaNet3D(self.training_config.model_config).to(device=self.training_config.gpu)
        return self

    def create_loss_function(self) -> Self:
        self.loss = TopologicalLoss(**asdict(self.training_config.loss_config))
        return self

    def create_optimizer(self) -> Self:
        dc = self.training_config.data_config
        self.optimizer = Ranger21(
            self.model.parameters(),
            lr=self.training_config.optimizer_config.lr,
            num_batches_per_epoch=(
                len(self.data_manager.train_dataset) // dc.train_batch_size +
                int(len(self.data_manager.train_dataset) % dc.train_batch_size != 0)),
            num_epochs=self.training_config.epochs,
            num_warmup_iterations=self.training_config.optimizer_config.num_warmup_iterations,
            logging_active=False,
        )
        return self

    # def create_scheduler_manager(self) -> Self:
    #     self.scheduler_manager = SchedulerManagerBuilder().build()
    #     return self

    def create_file_system_manager(self) -> Self:
        self.file_system_manager = FileSystemManager(self.training_config)
        return self

    def create_checkpoint_manager(self) -> Self:
        self.checkpoint_manager = CheckpointManager(
            ckpt_dir=self.training_config.checkpoint_root,
            mode="min",
            monitor="val_loss",
            top_k=self.training_config.save_top_k_checkpoints,
            top_k_best=self.training_config.save_top_k_best_checkpoints,
        )
        return self

    def create_tensorboard_logger(self) -> Self:
        self.tensorboard_logger = TensorBoardLogger(
            tensorboard_root=self.training_config.tensorboard_root,
        )
        return self

    def create_gradient_auto_clipper(self) -> Self:
        self.gradient_auto_clipper = GradientAutoClipper(
            model=self.model,
            unclipped_warmup_steps=0, #self.training_config.gradient_autoclip_unclipped_warmup_steps,
            percentile=0.1, #self.training_config.gradient_autoclip_percentile,
        )
        return self

    # def create_lr_scheduler(self) -> Self:
    #     self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         self.optimizer,
    #         T_max=self.training_config.epochs,
    #         eta_min=self.training_config.optimizer_config.lr_scheduler_eta_min,
    #     )
    #     return self

    def create_modules(self) -> Self:
        return (self
         .create_data_manager()
         .create_model()
         .create_loss_function()
         .create_optimizer()
         # .create_scheduler_manager()
         .create_file_system_manager()
         .create_checkpoint_manager()
         .create_tensorboard_logger()
         .create_gradient_auto_clipper()
         # .create_lr_scheduler()
        )

    def build_training_loop(self) -> TrainingLoop:
        return TrainingLoop(
            config=self.training_config,
            data_manager=self.data_manager,
            model=self.model,
            loss=self.loss,
            optimizer=self.optimizer,
            # scheduler_manager=self.scheduler_manager,
            file_system_manager=self.file_system_manager,
            checkpoint_manager=self.checkpoint_manager,
            tensorboard_logger=self.tensorboard_logger,
            gradient_auto_clipper=self.gradient_auto_clipper,
            # lr_scheduler=self.lr_scheduler,
        )
