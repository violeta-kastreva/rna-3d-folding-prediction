import plotly.graph_objects as go
import torch
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class RNAPlotter:
    CAMERA_DIRS = [
        dict(eye=dict(x=normed_vec[0], y=normed_vec[1], z=normed_vec[2]))
        for vec in (
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        for normed_vec in (np.array(vec) / np.linalg.norm(vec),)
    ]
    FONT_FILEPATH = "C:/Windows/Fonts/arialbd.ttf"

    def __init__(
        self,
        width: int = 400,
        height: int = 400,
        camera_distance_scale: float = 1.5,
        opacity: float = 0.1,
    ):
        self.width: int = width
        self.height: int = height
        self.camera_distance_scale: float = camera_distance_scale
        self.opacity: float = opacity

    def plot_sequence(
        self,
        target_id: str,
        sequence: str,
        predicted_coordinates: torch.Tensor,
        ground_truth_coordinates: torch.Tensor,
        do_plot_between_distances: bool = False,
        get_multiple_views: bool = True,
    ) -> np.ndarray:
        predicted: dict[str, np.ndarray] = self._parse_coordinates(predicted_coordinates)
        ground_truth: dict[str, np.ndarray] = self._parse_coordinates(ground_truth_coordinates)

        fig = go.Figure()

        if do_plot_between_distances:
            self._plot_between_distances(fig, predicted, ground_truth)

        self._plot_rna(fig, predicted, sequence, line_color="blue", marker_color="red")
        self._plot_rna(fig, ground_truth, sequence, line_color="green", marker_color="orange")

        fig.update_layout(
            title=target_id,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            width=self.width,
            height=self.height,
        )
        camera_dirs = [dict()] + (self.CAMERA_DIRS if get_multiple_views else [])
        images = np.zeros(shape=(len(camera_dirs), self.height, self.width, 3), dtype=np.uint8)
        for i, direction in enumerate(camera_dirs):
            if len(direction) > 0:
                camera = self._calculate_camera_params(predicted, ground_truth, direction)
                fig.update_layout(scene_camera=camera)

            images[i, ...] = self._get_image_from_figure(fig, text)

        if not get_multiple_views:
            images = images.squeeze(axis=0)

        return images

    def _parse_coordinates(self, coordinates: torch.Tensor) -> dict[str, np.ndarray]:
        coordinates = coordinates.cpu().numpy()
        return {
            c: coordinates[:, i]
            for i, c in enumerate(("x", "y", "z"))
        }

    def _plot_rna(
        self,
        fig: go.Figure,
        coords: dict[str, np.ndarray],
        sequence: str,
        line_color: str,
        marker_color: str,
    ):
        # Add lines connecting the nucleotides
        fig.add_trace(go.Scatter3d(
            **coords,
            mode="lines",
            line=dict(color=line_color, width=6),
            opacity=self.opacity,
        ))

        # Add spherical markers at each nucleotide position
        fig.add_trace(go.Scatter3d(
            **coords,
            mode="markers",
            marker=dict(
                size=5,
                color=marker_color,
                symbol="circle"
            ),
            text=list(sequence),
            opacity=self.opacity,
        ))

    def _plot_between_distances(
        self,
        fig: go.Figure,
        predicted: dict[str, np.ndarray],
        ground_truth: dict[str, np.ndarray],
    ):
        line_segments = {
            key: np.array([
                coord
                for y, y_hat in zip(predicted[key], ground_truth[key])
                for coord in (y, y_hat, None)
            ])
            for key in predicted.keys()
        }

        fig.add_trace(go.Scatter3d(
            **line_segments,
            mode="lines",
            line=dict(color="pink", width=4),
            opacity=self.opacity,
        ))

    def _calculate_camera_params(
        self,
        y: dict[str, np.ndarray],
        y_hat: dict[str, np.ndarray],
        dir: dict[str, float],
    ):
        points = {key: (y[key], y_hat[key]) for key in y.keys()}
        center = {
            key: (max(point.max() for point in points[key]) +
                  min(point.min() for point in points[key])) / 2
            for key in points.keys()
        }
        max_range: float = max(
            max(point.max() for point in points[key]) - min(point.min() for point in points[key])
            for key in points.keys()
        )

        camera_distance: float = self.camera_distance_scale * max_range

        camera = dict(
            eye={
                key: center[key] + camera_distance * dir[key]
                for key in center.keys()
            },
            center=center,
        )

        return camera

    def _get_image_from_figure(self, fig: go.Figure, text: str) -> np.ndarray:
        """
        Render the Plotly figure to an in-memory PNG image.
        """
        img_bytes = fig.to_image(format="png")
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        return self._write_text_on(image, text)

    def _write_text_on(self, image: Image, text: str) -> np.ndarray:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.FONT_FILEPATH, size=14)

        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_height = text_bbox[3] - text_bbox[1]

        # Position near bottom-left with padding
        x = 5
        y = self.height - text_height - x
        draw.text((x, y), text, fill="red", font=font, stroke_width=1, stroke_fill="black")
        return np.array(draw.im)
