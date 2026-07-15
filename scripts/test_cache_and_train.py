"""Small end-to-end test: generate cache for a small metadata subset and run one training epoch.

Usage: python scripts/test_cache_and_train.py
"""
from pathlib import Path
import tempfile
import shutil
import torch
from torch.utils.data import DataLoader

from config.base import get_config
from cache_system.generator import CacheGenerator
from metadata.patient_manifest import PatientManifestBuilder
from metadata.builder import MetadataBuilder
from datasets.cached_dataset import CachedMammoDataset
from models.simple_fusion import SimpleFusionModel


def main():
    cfg = get_config()

    # Ensure metadata exists
    if not cfg.metadata_csv.exists():
        print(f"Metadata CSV not found: {cfg.metadata_csv}")
        print("Attempting to scan 'abnormal' and 'normal' directories to build metadata")
        scanner = MetadataBuilder()
        # Try common locations
        for candidate in [Path("abnormal"), Path("normal")]:
            if candidate.exists():
                md = scanner.scan_directory(candidate, output_csv=cfg.metadata_csv)
                print(f"Wrote metadata to {cfg.metadata_csv} with {len(md)} rows")
                break
        else:
            print("No DICOM sources found; aborting test")
            return

    # Create a small metadata subset to speed up cache generation
    import pandas as pd

    df = pd.read_csv(cfg.metadata_csv)
    if df.empty:
        print("Metadata CSV is empty; aborting")
        return

    small_md = df.head(16).copy()
    small_csv = cfg.output_dir / "metadata_small.csv"
    small_md.to_csv(small_csv, index=False)

    # Generate cache for small subset
    gen = CacheGenerator(cfg.preprocessing)
    success, failed = gen.generate_from_metadata(small_csv, cfg.preprocessing.image_size)
    print(f"Cache generation: {success} successful, {failed} failed")

    # Build patient manifest if missing
    if not cfg.patient_manifest_csv.exists():
        pm = PatientManifestBuilder(cfg)
        pm.build_manifest(cfg.metadata_csv, output_csv=cfg.patient_manifest_csv)
        print(f"Built patient manifest at {cfg.patient_manifest_csv}")

    # Use a tiny manifest subset for training
    manifest_df = pd.read_csv(cfg.patient_manifest_csv)
    small_manifest = manifest_df.head(8)
    tmp_manifest = cfg.output_dir / "manifest_small.csv"
    small_manifest.to_csv(tmp_manifest, index=False)

    # Dataset + dataloader
    ds = CachedMammoDataset(tmp_manifest, cfg=cfg)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFusionModel(num_views=4, feat_dim=64, num_classes=2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for batch_idx, batch in enumerate(dl):
        imgs = batch["images"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(imgs, mask=mask)
        loss = loss_fn(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Batch {batch_idx}: loss={loss.item():.4f}")
        if batch_idx >= 2:
            break

    print("Test training run complete — model updated on cached data")


if __name__ == "__main__":
    main()
