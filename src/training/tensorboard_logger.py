import os
import time
import subprocess
from typing import cast, Union, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data.typing import DataBatch
from model.ribonanza_net_3d import RibonanzaNet3D, ModelConfig
from training.rna_plotter import RNAPlotter


class TensorBoardLogger:
    def __init__(self, tensorboard_root: str, port: int = 6006):
        self.tensorboard_root: str = tensorboard_root
        self.port: int = port

        self.writer: Optional[SummaryWriter] = None
        self.tensorboard_process: Optional[subprocess.Popen] = None

        self.rna_plotter: RNAPlotter = RNAPlotter()

    def setup_tensorboard(self, model: RibonanzaNet3D, device):
        tensorboard_path, file_existed = self.get_filepath()
        writer = SummaryWriter(tensorboard_path)

        # if not file_existed:
        #     self.graph_model(writer, model, device)

        # Start TensorBoard in the background
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", self.tensorboard_root, "--port", str(self.port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Give TensorBoard some time to start
        time.sleep(3)
        print(f"TensorBoard is running at: http://localhost:{self.port}/")

        self.writer = writer
        self.tensorboard_process = tensorboard_process

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
            input_data: torch.Tensor = cast(torch.Tensor, self._create_dummy_input(device, model.config))
            writer.add_graph(model, input_data)

    def add_scalar(self, tag: str, value: Union[float, torch.Tensor], step: int, as_tensor: bool = False):
        self.writer.add_scalar(tag, value, step, new_style=as_tensor)

    def add_scalars(self, tag: str, scalar_dict: dict[str, float], step: int):
        self.writer.add_scalars(tag, scalar_dict, step)

    def add_histogram(self, tag: str, values: torch.Tensor, step: int):
        self.writer.add_histogram(tag, values, step)

    def add_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        self.writer.add_image(tag, img_tensor, step)

    def add_text(self, tag: str, text_string: str, step: int):
        self.writer.add_text(tag, text_string, step)

    def add_rna_plot(
            self,
            tag: str,
            step: int,
            target_id: str,
            sequence: str,
            predicted_coordinates: torch.Tensor,
            ground_truth_coordinates: torch.Tensor,
            do_plot_between_distances: bool = True,
            get_multiple_views: bool = True,
    ):
        """
        Adds a RNA structure plot to TensorBoard.

        :param tag: The tag for the plot in TensorBoard.
        :param rna_sequence: The RNA sequence to plot.
        :param step: The current training step.
        """
        plot_image = self.rna_plotter.plot_sequence(
            target_id,
            sequence=sequence,
            predicted_coordinates=predicted_coordinates,
            ground_truth_coordinates=ground_truth_coordinates,
            do_plot_between_distances=do_plot_between_distances,
            get_multiple_views=get_multiple_views,
        )
        if get_multiple_views:
            self.writer.add_images(tag, plot_image, step, dataformats="NHWC")
        else:
            self.writer.add_image(tag, plot_image, step, dataformats='HWC')

    def add_rna_plots(
            self, tag: str,
            step: int,
            target_ids: list[str],
            sequences: list[str],
            predicted_coordinates: torch.Tensor,
            ground_truth_coordinates: torch.Tensor,
            lengths: torch.Tensor,
            do_plot_between_distances: bool = True,
    ):
        """
        Adds a RNA structure plot to TensorBoard.

        :param tag: The tag for the plot in TensorBoard.
        :param rna_sequence: The RNA sequence to plot.
        :param step: The current training step.
        """

        plot_images = np.stack([
            self.rna_plotter.plot_sequence(
            target_ids[i],
            sequence=sequences[i],
            predicted_coordinates=predicted_coordinates[i, :L, :],
            ground_truth_coordinates=ground_truth_coordinates[i, :L, :],
            do_plot_between_distances=do_plot_between_distances,
            get_multiple_views=False,
        )
            for i, L in enumerate(lengths)
        ], axis=0)

        self.writer.add_images(tag, plot_images, step, dataformats="NHWC")

    def close(self):
        self.writer.close()
        print("Should I close TensorBoard? (y/n)")
        input()
        if self.tensorboard_process:
            self.tensorboard_process.terminate()
            print("TensorBoard closed.")

    def _create_dummy_input(self, device: torch.device, config: ModelConfig) -> DataBatch:
        kwargs = dict(device=device, requires_grad=False)
        B: int = 1
        L: int = 3
        num_product_sequences: int = 2

        dummy_input: DataBatch = {
            "sequence": torch.zeros((B, L), dtype=torch.int32, **kwargs),  # Example sequence
            "msa": torch.zeros((B, L, config.d_msa), dtype=torch.int32, **kwargs),
            "msa_profiles": torch.zeros((B, L, config.d_msa_extra), dtype=torch.float32, **kwargs),
            "product_sequences": [
                torch.zeros((L,), dtype=torch.int32, **kwargs)
                for _ in range(num_product_sequences)
            ],
            "product_sequences_indices": torch.tensor(
                [
                    [i, j]
                    for i in range(B)
                    for j in range(num_product_sequences)
                ], dtype=torch.int32, **kwargs,
            ),
        }
        return dummy_input
