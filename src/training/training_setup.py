from dataclasses import asdict
from typing import Optional

from typing_extensions import Self

from data.data_manager import DataManager
from loss.topological_loss import TopologicalLoss
from model.ribonanza_net_3d import RibonanzaNet3D
from optimizer import Ranger21
from schedulers.scheduler_manager import SchedulerManager
from schedulers.scheduler_manager_builder import SchedulerManagerBuilder
from training.checkpoint_manager import CheckpointManager
from training.file_system_manager import FileSystemManager
from training.training_config import TrainingConfig
from training.training_loop import TrainingLoop


class TrainingSetup:
    """ TrainingLoop builder class."""
    def __init__(self, training_config: TrainingConfig):
        self.training_config: TrainingConfig = training_config
        self.data_manager: Optional[DataManager] = None  # Placeholder for DataManager type
        self.model: Optional[RibonanzaNet3D] = None
        self.optimizer: Optional[Ranger21] = None
        self.scheduler_manager: Optional[SchedulerManager] = None
        self.loss_function: Optional[TopologicalLoss] = None
        self.file_system_manager: Optional[FileSystemManager] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None

    def create_data_manager(self) -> DataManager:
        self.data_manager = DataManager(self.training_config.data_config, device=self.training_config.cpu)
        return self.data_manager

    def create_model(self) -> RibonanzaNet3D:
        self.model = RibonanzaNet3D(self.training_config.model_config).to(device=self.training_config.gpu)
        return self.model

    def create_loss_function(self) -> TopologicalLoss:
        self.loss_function = TopologicalLoss(**asdict(self.training_config.loss_config))
        return self.loss_function

    def create_optimizer(self):
        self.optimizer = Ranger21(self.model.parameters(), lr=self.training_config.optimizer_config.lr)
        return self.optimizer

    def create_scheduler_manager(self) -> SchedulerManager:
        return SchedulerManagerBuilder().build()

    def create_file_system_manager(self) -> FileSystemManager:
        self.file_system_manager = FileSystemManager(self.training_config)
        return self.file_system_manager

    def create_checkpoint_manager(self) -> CheckpointManager:
        self.checkpoint_manager = CheckpointManager(
            ckpt_dir=self.training_config.checkpoint_root,
            mode="min",
            monitor="val_loss",
            top_k=self.training_config.save_top_k_checkpoints,
            top_k_best=self.training_config.save_top_k_best_checkpoints,
        )
        return self.checkpoint_manager

    def create_modules(self) -> Self:
        self.create_data_manager()
        self.create_model()
        self.create_loss_function()
        self.create_optimizer()
        self.create_scheduler_manager()
        self.create_file_system_manager()
        self.create_checkpoint_manager()

        return self

    def build_training_loop(self) -> TrainingLoop:
        return TrainingLoop(
            config=self.training_config,
            data_manager=self.data_manager,
            model=self.model,
            loss=self.create_loss_function(),
            optimizer=self.create_optimizer(),
            scheduler_manager=self.scheduler_manager,
            file_system_manager=self.create_file_system_manager(),
            checkpoint_manager=self.create_checkpoint_manager(),
        )
