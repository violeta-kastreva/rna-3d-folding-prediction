from dataclasses import dataclass, field
from pathlib import Path

import torch

from data.data_manager import DataManagerConfig
from data.msa.msa_dataset import MSAConfig
from data.token_library import TokenLibrary
from loss.topological_loss import LossConfig
from model.ribonanza_net_3d import ModelConfig
from optimizer.optimizer_config import OptimizerConfig
from training.utils import join


source_code_root: Path = Path(__file__).parent.parent.resolve()

experimental_data_root: Path = Path(r"E:\Raw Datasets\Stanford RNA Dataset")
synthetic_data_root: Path = experimental_data_root / "converted"

training_directory: Path = Path(r"E:\train\RibonanzaNet3D\2025_06_25_Initial")

d_msa: int = 64


@dataclass
class TrainingConfig:
    cpu: torch.device = torch.device("cpu")
    gpu: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs: int = 50

    training_root: str = training_directory.as_posix()
    checkpoint_root: str = (training_directory / "checkpoints").as_posix()
    tensorboard_root: str = (training_directory / "tensorboard").as_posix()
    config_files: list[str] = field(default_factory=lambda: \
        join(source_code_root, ("training_config.py", "training_setyp.py")))

    save_top_k_checkpoints: int = 10
    save_top_k_best_checkpoints: int = 4

    model_config: ModelConfig = ModelConfig(
        d_model=128,
        n_heads=8,
        d_pair_repr=32,
        d_lstm=32,
        num_lstm_layers=16,
        num_blocks=8,
        d_msa=d_msa,
        use_triangular_attention=True,
        use_bidirectional_lstm=True,
        dropout=0.1,
        d_regr_outputs=3,
        d_prob_outputs=1,
        rel_pos_enc_clip_value=16,

        d_hidden=None,
        use_gradient_checkpoint=False,

        token_library=TokenLibrary(),
    )

    loss_config: LossConfig = LossConfig(
        rmse_weight = 0.4,
        cross_distance_weight=0.175,
        folding_angle_weight=0.175,
        probability_weight=0.25,
        probability_temperature=1.0,
        eps=1e-6,
    )

    optimizer_config: OptimizerConfig = OptimizerConfig(
        lr=0.001,
        clip_gradient_value=5.0,
    )

    data_config: DataManagerConfig = DataManagerConfig(
        train_sequence_files=join(experimental_data_root, ("train_sequences.csv", "train_sequences.v2.csv",)),
        train_label_files=join(experimental_data_root, ("train_sequences.csv", "train_sequences.v2.csv",)),

        val_sequence_files=join(experimental_data_root, ("validation_sequences.csv",)),
        val_label_files=join(experimental_data_root, ("validation_labels.csv",)),

        test_sequence_files=join(experimental_data_root, ("validation_sequences.csv",)),
        test_label_files=join(experimental_data_root, ("validation_labels.csv",)),

        msa_folders=join(experimental_data_root, ("MSA", "MSA_v2",)),
        msa_config=MSAConfig(
            block_size_remove_factor=0.15,
            num_blocks_to_remove=3,
            min_num_seqs_to_keep=10,
            num_representatives=d_msa,
            mutation_percent=0.15,
        ),

        combine_synthetic_with_real_data=False,
        synthetic_data_root_path=synthetic_data_root.as_posix(),
        synthetic_data_index_filepath=(synthetic_data_root / "index.csv").as_posix(),

        train_batch_size=4,
        val_batch_size=None,
        test_batch_size=None,

        train_num_workers=4,
        val_num_workers=2,
        test_num_workers=2,

        prefetch_factor=3,

        train_shuffle_radius=50,
        chance_flip_sequences=0.5,
        chance_use_msa_when_available=0.95,
    )
