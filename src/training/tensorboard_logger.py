import os
import time
import subprocess

import torch
from torch.utils.tensorboard import SummaryWriter

from data.typing import DataBatch
from model.ribonanza_net_3d import RibonanzaNet3D, ModelConfig


class TensorBoardLogger:
    def __init__(self, tensorboard_root: str, port: int = 6006):
        self.tensorboard_root: str = tensorboard_root
        self.port: int = port

        self.writer, self.tensorboard_process = self.setup_tensorboard()

    def setup_tensorboard(self, model: RibonanzaNet3D, device) -> tuple[SummaryWriter, subprocess.Popen]:
        tensorboard_path, file_existed = self.get_filepath()
        writer = SummaryWriter(tensorboard_path)

        if not file_existed:
            self.graph_model(writer, model, device)

        # Start TensorBoard in the background
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", self.tensorboard_root, "--port", str(self.port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Give TensorBoard some time to start
        time.sleep(3)
        print(f"TensorBoard is running at: http://localhost:{self.port}/")
        return writer, tensorboard_process

    def get_filepath(self) -> tuple[str, bool]:
        files: list[str] = os.listdir(self.tensorboard_root)
        file_existed: bool = len(files) > 0
        if file_existed:
            tensorboard_path: str = files[0]
        else:
            tensorboard_path: str = os.path.join(
                self.tensorboard_root,
                f"experiment_{time.strftime('%Y-%m-%d_%H-%M-%S')}.tb",
            )
        return tensorboard_path, file_existed

    def graph_model(self, writer: SummaryWriter, model: RibonanzaNet3D, device: torch.device):
        with torch.no_grad():
            writer.add_graph(model, self._create_dummy_input(device, model.config))

    def _create_dummy_input(self, device: torch.device, config: ModelConfig) -> DataBatch:

        dummy_input: DataBatch = { # TODO: Fix
            "sequence": torch.randint(1, 10, size=(1, 100), dtype=torch.int64, device=device, requires_grad=False),  # Example sequence
            "msa": torch.randn(1, 100, 64),  # Example MSA with 64 features
            "product_sequences": torch.randn(1, 100, 100, 32),  # Example pair representation
        }
        return dummy_input
