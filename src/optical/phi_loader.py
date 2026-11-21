from __future__ import annotations

import math
from pathlib import Path

import torch

from src.paths import ASSETS, THIRD_PARTY

SIMULATION_256 = THIRD_PARTY / "latent-imaging" / "paper_checkpoints" / "simulation_256.pt"
MASKING_PY = THIRD_PARTY / "latent-imaging" / "models" / "masking.py"


def binarize_lsi_weight(weight: torch.Tensor, bits: int = 1) -> torch.Tensor:
    """Apply the LSI detach quantizer and return Phi as (M, H*W).

    masking.py stores weight as (latent_size, H*W), permutes to (H*W, M),
    clamps to [0, 1], then rounds with b = 2**bits - 1. bits=1 is binary.
    """
    if weight.ndim != 2:
        raise ValueError(f"mask weight must be 2D, got {tuple(weight.shape)}")
    real = weight.detach().float().permute(1, 0).clamp(0, 1)
    levels = (2**int(bits)) - 1
    if levels <= 0:
        raise ValueError(bits)
    quantized = (real * levels).round() / levels
    return quantized.transpose(0, 1).contiguous()


def is_binary_01(weight: torch.Tensor, tol: float = 1e-5) -> bool:
    flat = weight.detach().float().reshape(-1)
    if flat.numel() == 0:
        return False
    close = ((flat - 0).abs() <= tol) | ((flat - 1).abs() <= tol)
    return bool(torch.all(close).item())


def official_quantized_mask(weight: torch.Tensor, bits: int = 1) -> torch.Tensor:
    """Exact bits quantization from masking.py quantizing_detach, without the STE add-back.

    Checkpoint layout is (M, H*W). The module permutes to (H*W, M), clamps to
    [0, 1], and applies b = 2**bits - 1. Returns that quantized matrix as (M, H*W).
    """
    return binarize_lsi_weight(weight, bits=bits)


def official_rgb_mask_multiply(image_01: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Official sum-of-RGB then one shared mask: (R + G + B) @ W.

    image_01 is (B, 3, H, W) in [0, 1] at Phi resolution. phi is (M, H*W).
    masking.py writes the same product as three channel matmuls.
    """
    if image_01.ndim != 4 or int(image_01.shape[1]) != 3:
        raise ValueError(f"expected BCHW RGB, got {tuple(image_01.shape)}")
    summed = image_01[:, 0] + image_01[:, 1] + image_01[:, 2]
    flat = summed.reshape(summed.shape[0], -1)
    return flat @ phi.transpose(0, 1).to(dtype=flat.dtype)


def official_channel_mask_multiply(image_01: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """masking.py forward association: (R @ W) + (G @ W) + (B @ W)."""
    if image_01.ndim != 4 or int(image_01.shape[1]) != 3:
        raise ValueError(f"expected BCHW RGB, got {tuple(image_01.shape)}")
    weight = phi.transpose(0, 1).to(dtype=image_01.dtype)
    total = None
    for channel in range(3):
        flat = image_01[:, channel].reshape(image_01.shape[0], -1)
        term = flat @ weight
        total = term if total is None else total + term
    return total


def read_checkpoint_weight(path: Path) -> tuple[torch.Tensor, str]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    weight, key = _extract_weight(payload)
    return weight.detach().cpu(), key


def _stub_matplotlib_if_needed() -> None:
    import sys
    import types

    try:
        import matplotlib.pyplot  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    mpl = sys.modules.get("matplotlib")
    if mpl is None:
        mpl = types.ModuleType("matplotlib")
        sys.modules["matplotlib"] = mpl
    if not hasattr(mpl, "use"):
        mpl.use = lambda *args, **kwargs: None
    if "matplotlib.pyplot" not in sys.modules:
        sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")


def load_mask_generator_class():
    """Load MaskGenerator from the cloned latent-imaging repo."""
    import importlib.util
    import sys

    module_name = "physflow_lsi_masking"
    if module_name in sys.modules:
        return sys.modules[module_name].MaskGenerator
    if not MASKING_PY.is_file():
        raise FileNotFoundError(MASKING_PY)
    _stub_matplotlib_if_needed()
    spec = importlib.util.spec_from_file_location(module_name, MASKING_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MASKING_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module.MaskGenerator


def official_module_forward(image_01: torch.Tensor, raw_weight: torch.Tensor, bits: int = 1) -> dict:
    """Run MaskGenerator.forward on a [-1, 1] copy of a [0, 1] RGB image.

    Dropout is off and the module is in eval mode, so the mask is the
    bits=1 detach quantizer. Returns the measurement and the STE weight.
    """
    from types import SimpleNamespace

    mask_cls = load_mask_generator_class()
    raw = raw_weight.detach().float().cpu()
    args = SimpleNamespace(
        clip_w="clamp",
        encoding_size=int(raw.shape[0]),
        quantizing_strategy="detach",
        verbose=False,
        bits=int(bits),
        w_dropout=0.0,
        dropout=0.0,
        ini_w="normalized",
    )
    generator = mask_cls(int(raw.shape[1]), args)
    with torch.no_grad():
        generator.weight.copy_(raw)
        generator.eval()
        # masking.py maps its incoming tensor from [-1, 1] into [0, 1] before the product.
        measurement, ste_weight = generator(image_01.detach().float().cpu() * 2.0 - 1.0, 0)
    return {
        "measurement": measurement.detach().cpu(),
        "ste_weight": ste_weight.detach().cpu(),
    }


def assert_official_phi(raw_weight: torch.Tensor, bits: int = 1) -> dict:
    """Raw checkpoint weights are not {0, 1}; bits=1 quantization is; measurement matches.

    The forward check is a [0, 1] RGB image at Phi resolution against
    sum-of-RGB then mask multiply, tolerance 1e-5, in float64.
    """
    from src.optical.spi_operator import SPIOperator

    raw = raw_weight.detach().cpu()
    raw_is_binary = is_binary_01(raw)
    phi = binarize_lsi_weight(raw, bits=bits)
    phi_is_binary = is_binary_01(phi, tol=0.0)
    side = int(math.isqrt(int(phi.shape[1])))
    report = {
        "raw_shape": list(raw.shape),
        "raw_dtype": str(raw.dtype).replace("torch.", ""),
        "raw_is_binary_01": raw_is_binary,
        "binarized_is_binary_01": phi_is_binary,
        "phi_shape": list(phi.shape),
        "resolution": side if side * side == int(phi.shape[1]) else None,
        "bits": int(bits),
        "fraction_ones": float((phi > 0.5).float().mean()) if phi.numel() else None,
        "raw_min": float(raw.float().min()) if raw.numel() else None,
        "raw_max": float(raw.float().max()) if raw.numel() else None,
    }
    if raw_is_binary:
        report["error"] = "loaded raw mask weight is already {0,1}"
        raise AssertionError(report["error"])
    if side * side != int(phi.shape[1]):
        report["error"] = f"Phi pixel count {int(phi.shape[1])} is not a square"
        raise AssertionError(report["error"])
    if not phi_is_binary:
        report["error"] = "binarized Phi is not exactly {0,1}"
        raise AssertionError(report["error"])
    if not bool(((phi == 0).any() and (phi == 1).any()).item()):
        report["error"] = "binarized Phi does not contain both 0 and 1"
        raise AssertionError(report["error"])

    generator = torch.Generator().manual_seed(0)
    image = torch.rand(2, 3, side, side, generator=generator)
    phi64 = phi.double()
    image64 = image.double()
    operator = SPIOperator(phi)
    measured = operator.measure(image64, input_range="0_1")
    summed = official_rgb_mask_multiply(image64, phi64)
    channels = official_channel_mask_multiply(image64, phi64)
    sum_err = float((measured - summed).abs().max())
    channel_err = float((measured - channels).abs().max())
    report["official_forward_max_abs"] = sum_err
    report["official_channel_sum_max_abs"] = channel_err
    if sum_err >= 1e-5:
        report["error"] = f"sum-of-RGB mask multiply max abs {sum_err} exceeds 1e-5"
        raise AssertionError(report["error"])
    if channel_err >= 1e-5:
        report["error"] = f"per-channel mask multiply max abs {channel_err} exceeds 1e-5"
        raise AssertionError(report["error"])

    module = official_module_forward(image, raw, bits=bits)
    ste = module["ste_weight"]
    if ste.shape != (side * side, int(phi.shape[0])):
        report["error"] = f"official STE weight shape {tuple(ste.shape)}"
        raise AssertionError(report["error"])
    ste_gap = float((ste - phi.transpose(0, 1)).abs().max())
    # masking.py converts its input from [-1, 1] to [0, 1] before the product.
    # Compare the module to that same converted image, in float32, not to a float64 sum.
    roundtrip = (image.float() * 2.0 - 1.0) * 0.5 + 0.5
    module_ref = official_channel_mask_multiply(roundtrip, phi.float())
    module_gap = float((module["measurement"] - module_ref).abs().max())
    report["official_ste_weight_max_abs"] = ste_gap
    report["official_module_forward_max_abs"] = module_gap
    if ste_gap >= 1e-5:
        report["error"] = f"official MaskGenerator mask differs from bits=1 Phi by {ste_gap}"
        raise AssertionError(report["error"])
    if module_gap >= 1e-5:
        report["error"] = f"official MaskGenerator measurement differs from mask multiply by {module_gap}"
        raise AssertionError(report["error"])
    report["official_binarizer_matches"] = True
    report["phi"] = phi
    return report


def _candidate_rank(path: Path) -> tuple:
    name = path.name.lower()
    if "simulation" in name:
        group = 0
    elif "256" in name:
        group = 1
    elif "128" in name:
        group = 2
    elif "32" in name:
        group = 3
    else:
        group = 4
    return (group, name)


def _candidate_files() -> list[Path]:
    roots = [
        THIRD_PARTY / "latent-imaging" / "paper_checkpoints",
        ASSETS / "lsi",
        ASSETS,
    ]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in {".pt", ".pth", ".ckpt"} and path.is_file():
                found.append(path)
    found.sort(key=_candidate_rank)
    return found


def _extract_weight(payload) -> tuple[torch.Tensor, str]:
    if isinstance(payload, torch.Tensor):
        return payload, "tensor"
    if not isinstance(payload, dict):
        raise TypeError(type(payload).__name__)
    if "mask" in payload:
        mask = payload["mask"]
        if isinstance(mask, dict) and "weight" in mask:
            return mask["weight"], "ckpt['mask']['weight']"
        if isinstance(mask, torch.Tensor):
            return mask, "ckpt['mask']"
    for key in ("weight", "phi", "mask_weight"):
        if key in payload and isinstance(payload[key], torch.Tensor):
            return payload[key], key
    state = payload.get("state_dict", payload)
    if isinstance(state, dict):
        preferred = []
        others = []
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                continue
            if "mask" in key.lower() and key.endswith("weight"):
                preferred.append((key, value))
            else:
                others.append((key, value))
        if preferred:
            key, value = max(preferred, key=lambda item: item[1].numel())
            return value, f"state_dict:{key}"
        squareish = [(k, v) for k, v in others if _looks_like_phi(v)]
        if squareish:
            key, value = max(squareish, key=lambda item: item[1].numel())
            return value, f"state_dict:{key}"
    raise KeyError("no mask weight found in checkpoint")


def _looks_like_phi(tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2:
        return False
    a, b = tensor.shape
    sides = {int(math.isqrt(int(a))), int(math.isqrt(int(b)))}
    return any(side * side in (int(a), int(b)) and side >= 8 for side in sides)


def load_phi_checkpoint(path: Path, bits: int = 1) -> dict:
    weight, key = read_checkpoint_weight(path)
    phi = binarize_lsi_weight(weight, bits=bits)
    stored = weight.detach().cpu()
    raw = stored.float()
    flat = raw.reshape(-1)
    n_unique = None
    unique_preview = None
    if 0 < flat.numel() <= 20_000_000:
        uniq = torch.unique(flat)
        n_unique = int(uniq.numel())
        if n_unique <= 12:
            unique_preview = [float(value) for value in uniq]
    near_binary = ((flat - 0).abs() < 1e-4) | ((flat - 1).abs() < 1e-4)
    if stored.ndim == 2:
        tensor_layout = "2D matrix. One shared spatial pattern, no RGB axis in the tensor."
    elif stored.ndim == 3 and int(stored.shape[0]) == 3:
        tensor_layout = "3D tensor whose leading axis is 3."
    else:
        tensor_layout = f"{int(stored.ndim)}D tensor."
    unique = torch.unique(phi.flatten()).tolist()
    side = int(math.isqrt(int(phi.shape[1])))
    return {
        "path": str(path),
        "key": key,
        "raw_dtype": str(stored.dtype).replace("torch.", ""),
        "raw_ndim": int(stored.ndim),
        "raw_shape": list(raw.shape),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_n_unique": n_unique,
        "raw_unique_values": unique_preview,
        "fraction_in_unit_interval": float(((flat >= 0) & (flat <= 1)).float().mean()) if flat.numel() else None,
        "fraction_near_binary": float(near_binary.float().mean()) if flat.numel() else None,
        "tensor_layout": tensor_layout,
        "phi_shape": list(phi.shape),
        "resolution": side if side * side == int(phi.shape[1]) else None,
        "unique_after_binarize": unique,
        "fraction_ones": float((phi > 0.5).float().mean()),
        "phi": phi,
        "bits": bits,
        "color_rule": "sum of RGB in [0,1], shared mask, from models/masking.py",
    }


def search_phi_checkpoints() -> dict:
    files = _candidate_files()
    errors = []
    records = []
    chosen = None
    for path in files:
        try:
            loaded = load_phi_checkpoint(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        records.append({key: value for key, value in loaded.items() if key != "phi"})
        if chosen is None:
            chosen = loaded
        else:
            del loaded
    searched = [str(item) for item in files]
    if chosen is None:
        return {"phi": None, "searched": searched, "errors": errors, "records": records}
    chosen["searched"] = searched
    chosen["errors"] = errors
    chosen["records"] = records
    return chosen


def find_phi_checkpoint() -> dict | None:
    found = search_phi_checkpoints()
    if found.get("phi") is None:
        return None
    return found
