from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CGeo:
    """Canonical geometry container. `source` is insightface106 on day 1, smirk later."""

    vector: torch.Tensor
    landmarks: torch.Tensor | None = None
    source: str = "insightface106"

    @property
    def dim(self) -> int:
        return int(self.vector.shape[-1])


INSIGHTFACE_106_LEFT_EYE = list(range(66, 75))
INSIGHTFACE_106_RIGHT_EYE = list(range(75, 84))


def landmarks_to_vector(landmarks: torch.Tensor, image_size: int) -> torch.Tensor:
    flat = landmarks.reshape(landmarks.shape[0], -1).float() / float(image_size)
    if flat.shape[-1] == 212:
        return flat
    out = flat.new_zeros(flat.shape[0], 212)
    n = min(212, flat.shape[-1])
    out[:, :n] = flat[:, :n]
    return out


class InsightFaceBundle:
    """Lazy InsightFace teachers. None of these modules are imported by the solver."""

    def __init__(self, det_size: int = 256, root: str | None = None):
        self.det_size = det_size
        self.root = root
        self.errors: dict[str, str] = {}
        self.antelope = None
        self.buffalo = None
        self.adaface = None
        self.landmark_app = None

    def _app(self, name: str):
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            self.errors[name] = f"insightface import failed: {exc}"
            return None
        root_kw = {"root": self.root} if self.root else {}
        attempts = (
            (["CUDAExecutionProvider", "CPUExecutionProvider"], 0),
            (["CPUExecutionProvider"], -1),
        )
        errors = []
        for providers, ctx in attempts:
            try:
                app = FaceAnalysis(name=name, providers=providers, **root_kw)
                app.prepare(ctx_id=ctx, det_size=(self.det_size, self.det_size))
                return app
            except Exception as exc:
                errors.append(f"{providers[0]} ctx {ctx}: {exc}")
        self.errors[name] = " | ".join(errors)
        return None

    def load_antelope(self):
        if self.antelope is None:
            self.antelope = self._app("antelopev2")
        return self.antelope

    def load_buffalo(self):
        if self.buffalo is None:
            self.buffalo = self._app("buffalo_l")
        return self.buffalo

    def load_landmarks(self):
        if self.landmark_app is None:
            self.landmark_app = self._app("buffalo_l")
        return self.landmark_app

    def load_adaface(self):
        if self.adaface is not None or "adaface" in self.errors:
            return self.adaface
        try:
            from transformers import AutoModel
        except Exception as exc:
            self.errors["adaface"] = f"transformers unavailable: {exc}"
            return None
        try:
            self.adaface = AutoModel.from_pretrained("minchul/cvlface_adaface_ir101_webface4m", trust_remote_code=True)
        except Exception as exc:
            self.errors["adaface"] = str(exc)
            self.adaface = None
        return self.adaface


def _bgr_uint8(image: torch.Tensor) -> "object":
    import numpy as np

    x = image.detach().float().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if float(x.min()) < 0:
        x = x * 0.5 + 0.5
    array = (x.clamp(0, 1).numpy() * 255.0).astype("uint8")
    if array.shape[-1] == 3:
        array = array[:, :, ::-1].copy()
    return array


def embed_with_app(app, image: torch.Tensor) -> torch.Tensor | None:
    if app is None:
        return None
    import numpy as np

    if image.ndim == 3:
        image = image.unsqueeze(0)
    rows = []
    for sample in image:
        faces = app.get(_bgr_uint8(sample))
        if not faces:
            return None
        face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
        rows.append(torch.from_numpy(np.asarray(face.normed_embedding)).float())
    return torch.stack(rows, dim=0)


def landmarks_with_app(app, image: torch.Tensor) -> torch.Tensor | None:
    if app is None:
        return None
    import numpy as np

    if image.ndim == 3:
        image = image.unsqueeze(0)
    rows = []
    for sample in image:
        faces = app.get(_bgr_uint8(sample))
        if not faces or not hasattr(faces[0], "landmark_2d_106"):
            return None
        face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
        rows.append(torch.from_numpy(np.asarray(face.landmark_2d_106)).float())
    return torch.stack(rows, dim=0)


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (a * b).sum(dim=-1)
