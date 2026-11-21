"""Frozen AntelopeV2 glintr100, with gradients only into the image.

The ONNX graph is Conv, PRelu, BatchNormalization, Add, Flatten, and Gemm.
Parameters are buffers. A fixed affine maps the aligned 512 frame onto the
112 ArcFace crop, so the detector is not in the graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.align import invert_similarity, similarity_matrix, template_5

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def _initializer(tensor) -> torch.Tensor:
    from onnx.numpy_helper import to_array

    array = np.ascontiguousarray(to_array(tensor))
    return torch.from_numpy(array.copy())


def _ints(node, name: str, default: list[int]) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return default


def _float(node, name: str, default: float) -> float:
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return default


def _prelu(x: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    weight = slope.reshape(-1)
    if weight.numel() != 1 and weight.numel() != x.shape[1]:
        weight = slope.reshape(x.shape[1], -1)[:, 0].contiguous()
    return F.prelu(x, weight)


def _conv(node, values: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = [values[name] for name in node.input if name]
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None
    pads = _ints(node, "pads", [0, 0, 0, 0])
    strides = _ints(node, "strides", [1, 1])
    dilations = _ints(node, "dilations", [1, 1])
    group = 1
    for attr in node.attribute:
        if attr.name == "group":
            group = int(attr.i)
    if pads[0] == pads[2] and pads[1] == pads[3]:
        return F.conv2d(
            x,
            weight,
            bias,
            stride=strides,
            padding=(pads[0], pads[1]),
            dilation=dilations,
            groups=group,
        )
    x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
    return F.conv2d(x, weight, bias, stride=strides, padding=0, dilation=dilations, groups=group)


class GlintR100(nn.Module):
    def __init__(self, path: str | Path):
        super().__init__()
        import onnx

        model = onnx.load(str(path))
        graph = model.graph
        weights: dict[str, torch.Tensor] = {}
        for index, item in enumerate(graph.initializer):
            tensor = _initializer(item).float()
            buffer = f"b{index}"
            self.register_buffer(buffer, tensor)
            weights[item.name] = buffer
        self._weight_names = weights
        self._nodes = []
        for node in graph.node:
            self._nodes.append(
                (
                    node.op_type,
                    list(node.input),
                    list(node.output),
                    node,
                )
            )
        self.input_name = graph.input[0].name
        self.output_name = graph.output[0].name

    def _value(self, name: str, values: dict[str, torch.Tensor]) -> torch.Tensor:
        if name in values:
            return values[name]
        return getattr(self, self._weight_names[name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = {self.input_name: x.float()}
        for op, inputs, outputs, node in self._nodes:
            args = [self._value(name, values) for name in inputs if name]
            if op == "Conv":
                out = _conv(node, {name: self._value(name, values) for name in inputs if name})
            elif op == "PRelu":
                out = _prelu(args[0], args[1])
            elif op == "BatchNormalization":
                out = F.batch_norm(
                    args[0],
                    args[3],
                    args[4],
                    args[1],
                    args[2],
                    training=False,
                    eps=_float(node, "epsilon", 1e-5),
                )
            elif op == "Add":
                out = args[0] + args[1]
            elif op == "Flatten":
                axis = 1
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = int(attr.i)
                out = args[0].flatten(axis)
            elif op == "Gemm":
                weight = args[1]
                bias = args[2] if len(args) > 2 else None
                trans_b = 0
                for attr in node.attribute:
                    if attr.name == "transB":
                        trans_b = int(attr.i)
                if trans_b == 0:
                    weight = weight.transpose(0, 1)
                out = F.linear(args[0], weight, bias)
            else:
                raise NotImplementedError(op)
            values[outputs[0]] = out
        return values[self.output_name]


class FixedArcfaceCrop(nn.Module):
    """One 512-aligned frame to the 112 ArcFace template. Same matrix every image."""

    def __init__(self):
        super().__init__()
        matrix = similarity_matrix(template_5(512), ARCFACE_DST)
        inverse = invert_similarity(matrix)
        ys, xs = np.meshgrid(np.arange(112), np.arange(112), indexing="ij")
        ones = np.ones_like(xs, dtype=np.float64)
        dest = np.stack([xs.reshape(-1), ys.reshape(-1), ones.reshape(-1)], axis=1)
        src = dest @ inverse.T
        grid_x = src[:, 0] / 511.0 * 2.0 - 1.0
        grid_y = src[:, 1] / 511.0 * 2.0 - 1.0
        grid = np.stack([grid_x, grid_y], axis=1).reshape(1, 112, 112, 2).astype(np.float32)
        self.register_buffer("grid", torch.from_numpy(grid))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grid = self.grid.to(dtype=image.dtype, device=image.device).expand(image.shape[0], -1, -1, -1)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
