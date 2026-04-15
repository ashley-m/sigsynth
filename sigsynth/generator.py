from __future__ import annotations

import io
import inspect
from pathlib import Path
import zipfile

import numpy as np

from sigsynth.models import AppConfig
from sigsynth.registry import GENERATOR_REGISTRY


def _call_with_supported_kwargs(callable_obj, kwargs: dict):
    signature = inspect.signature(callable_obj)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(**supported)


def _build_torchsig_metadata(config: AppConfig):
    """Create the most compatible TorchSig metadata object available."""
    try:
        from torchsig.datasets import dataset_metadata as metadata_mod  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external submodule state
        raise RuntimeError("TorchSig metadata module is unavailable.") from exc

    generator_tags = {
        tag
        for name in config.generators
        for tag in GENERATOR_REGISTRY.get(name, GENERATOR_REGISTRY["BPSK"]).tags
    }
    dataset_type = "wideband" if "wideband" in generator_tags else "narrowband"

    sample_rate = int(config.global_params.get("sample_rate", 1_000_000))
    sample_len = int(config.global_params.get("sample_len", 1024))
    num_samples = int(config.dataset.total_samples)

    metadata_kwargs = {
        "dataset_type": dataset_type,
        "num_samples": num_samples,
        "sample_rate": sample_rate,
        "num_iq_samples_dataset": sample_len,
        "num_iq_samples": sample_len,
    }

    if hasattr(metadata_mod, "NarrowbandMetadata") and dataset_type == "narrowband":
        return _call_with_supported_kwargs(metadata_mod.NarrowbandMetadata, metadata_kwargs)
    if hasattr(metadata_mod, "WidebandMetadata") and dataset_type == "wideband":
        return _call_with_supported_kwargs(metadata_mod.WidebandMetadata, metadata_kwargs)
    if hasattr(metadata_mod, "DatasetMetadata"):
        return _call_with_supported_kwargs(metadata_mod.DatasetMetadata, metadata_kwargs)

    raise RuntimeError("No compatible TorchSig metadata constructor found.")


def _attempt_torchsig_generation(config: AppConfig, output_dir: Path) -> tuple[bool, str | None]:
    """Best-effort generation through TorchSig APIs with fallback behavior."""
    try:
        from torchsig.utils.generate import generate as torchsig_generate  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external submodule state
        return False, f"TorchSig generate API unavailable: {exc}"

    try:
        metadata = _build_torchsig_metadata(config)
        batch_size = max(1, min(256, int(config.dataset.total_samples)))
        torchsig_generate(
            root=str(output_dir),
            dataset_metadata=metadata,
            batch_size=batch_size,
            num_workers=0,
        )
        return True, None
    except Exception as exc:  # pragma: no cover - depends on torchsig version
        return False, str(exc)


def generate_dataset(config: AppConfig) -> dict[str, int | str | bool]:
    output_dir = Path(config.dataset.output_dir)
    train_count = int(config.dataset.total_samples * config.dataset.train_ratio)
    val_count = config.dataset.total_samples - train_count

    layout = [
        output_dir / "train" / "raw",
        output_dir / "train" / "impaired",
        output_dir / "val" / "raw",
        output_dir / "val" / "impaired",
    ]
    for folder in layout:
        folder.mkdir(parents=True, exist_ok=True)

    torchsig_generated, torchsig_error = _attempt_torchsig_generation(config, output_dir)

    rng = np.random.default_rng(seed=53)
    sample_len = int(config.global_params.get("sample_len", 1024))

    if not torchsig_generated:
        for split, count in (("train", train_count), ("val", val_count)):
            for idx in range(count):
                raw = rng.standard_normal(sample_len) + 1j * rng.standard_normal(sample_len)
                impaired = raw * (1 + 0.01 * rng.standard_normal(sample_len))

                np.save(output_dir / split / "raw" / f"sample_{idx:06d}.npy", raw)
                np.save(output_dir / split / "impaired" / f"sample_{idx:06d}.npy", impaired)

    return {
        "output_dir": str(output_dir),
        "train_samples": train_count,
        "val_samples": val_count,
        "torchsig_generated": torchsig_generated,
        "torchsig_error": torchsig_error or "",
    }


def build_dataset_zip_bytes(output_dir: str | Path) -> bytes:
    root = Path(output_dir)
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(root))
    memory_file.seek(0)
    return memory_file.getvalue()
