import os
import shutil

from training.training_config import TrainingConfig


class FileSystemManager:
    def __init__(self, config: TrainingConfig):
        self.config: TrainingConfig = config

    def ensure_created(self):
        for dir in (
            self.config.training_root,
            self.config.checkpoint_root,
            self.config.tensorboard_root,
            *self.config.data_config.msa_folders,
        ) + ((
            self.config.data_config.synthetic_data_root_path,
        ) if self.config.data_config.combine_synthetic_with_real_data else tuple()):
            self._ensure_created_directory(dir)

        for file in (
            *self.config.data_config.train_sequence_files,
            *self.config.data_config.train_label_files,
            *self.config.data_config.val_sequence_files,
            *self.config.data_config.val_label_files,
            *self.config.data_config.test_sequence_files,
            *self.config.data_config.test_label_files,
        ) + ((
            self.config.data_config.synthetic_data_index_filepath,
        ) if self.config.data_config.combine_synthetic_with_real_data else tuple()):
            self._ensure_file_exists(file)

        for file in self.config.config_files:
            self._ensure_file_is_copied(file)

    def _ensure_created_directory(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    def _ensure_file_exists(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")

    def _ensure_file_is_copied(self, filepath: str):
        new_filepath: str = os.path.join(self.config.training_root, os.path.basename(filepath))
        shutil.copy(filepath, new_filepath)
        print(f"Copied file to: {new_filepath}")