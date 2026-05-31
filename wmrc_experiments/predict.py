from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import WMRCGraphDataset, collate_graphs, record_from_path, save_predictions_csv, split_by_subject
from .models import build_model


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="WMRC_general")
    parser.add_argument("--task", default=None, help="task name or all; defaults to checkpoint task")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--input", default=None, help="single .pt file or directory of .pt files")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    task = args.task if args.task is not None else ckpt.get("task", "all")

    rows = []
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            files = sorted(input_path.glob("**/*.pt"))
        else:
            files = [input_path]
        records = [record_from_path(f, pe_dim=ckpt["config"]["pe_dim"]) for f in files]
        loader_data = {
            "x": torch.stack([r.x for r in records]),
            "adj": torch.stack([r.adj for r in records]),
            "pe": torch.stack([r.pe for r in records]),
            "comm": torch.stack([r.community_labels for r in records]),
            "y": torch.tensor([r.label for r in records]),
        }
        batches = [loader_data]
        meta_records = records
    else:
        dataset = WMRCGraphDataset(args.data_root, task_filter=task, pe_dim=ckpt["config"]["pe_dim"])
        split = split_by_subject(dataset, seed=ckpt.get("seed", 42))
        indices = split["indices"][args.split] if args.split != "all" else list(range(len(dataset)))
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collate_graphs)
        batches = list(loader)
        meta_records = [dataset.records[i] for i in indices]

    offset = 0
    with torch.no_grad():
        for batch in batches:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            pe = batch["pe"].to(device)
            comm = batch["comm"].to(device)
            mask = batch["mask"].to(device)
            logits = model(x, adj, pe=pe, comm=comm, mask=mask)
            probs = torch.softmax(logits, dim=1).cpu()
            pred = probs.argmax(dim=1)
            for i in range(probs.size(0)):
                record = meta_records[offset + i]
                rows.append(
                    {
                        "path": record.path,
                        "task": record.task,
                        "subject": record.subject,
                        "label": record.label,
                        "pred": int(pred[i].item()),
                        "prob_0": float(probs[i, 0].item()),
                        "prob_1": float(probs[i, 1].item()),
                    }
                )
            offset += probs.size(0)

    save_predictions_csv(rows, args.output)
    print(f"saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    predict_main()
