#!/usr/bin/env python3
"""
Train a PyTorch LSTM on input15m.txt using the current debug-run parameters inferred
from the user's uploaded training logs and the public ExpertAdvisor repo.

Matched settings:
- input columns: dt, open, close, high, low, vol, target
- engineered feature size: 14 (mirrors Tensor::Add in the repo)
- hidden size: 64
- target type: BinaryReturn (future close > current close)
- window_size: 64
- prediction_horizon: 4
- learning_rate: 1e-4
- load_latest: false / start from scratch
- default training range: 2010-01-01 to 2025-01-01

This script is intentionally standalone and avoids any project-specific DB code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


K_FEATURE_SCALE = 1000.0
ROLLING_VOL_LOOKBACK = 32
ROLLING_RET_LOOKBACK = 32
FEATURE_SIZE = 14
HIDDEN_SIZE = 64
DEFAULT_WINDOW_SIZE = 64
DEFAULT_PREDICTION_HORIZON = 4
DEFAULT_C_NEXT_THRESHOLD = 0.0012
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_START = "2010-01-01"
DEFAULT_TRAIN_END = "2025-01-01"
DEFAULT_VALID_FRACTION = 0.1
DEFAULT_SEED = 42


@dataclass
class TrainConfig:
    input_path: str
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    window_size: int = DEFAULT_WINDOW_SIZE
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON
    c_next_threshold: float = DEFAULT_C_NEXT_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    hidden_size: int = HIDDEN_SIZE
    valid_fraction: float = DEFAULT_VALID_FRACTION
    seed: int = DEFAULT_SEED
    device: str = "auto"
    output_dir: str = "training_output"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ForexSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


class BinaryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -0.01, 0.01)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.numel()
                # PyTorch LSTM gate order: i, f, g, o
                start = n // 4
                end = n // 2
                with torch.no_grad():
                    param[start:end].fill_(1.0)
        nn.init.constant_(self.head.weight, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class WarmupBinaryMetrics:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.count = 0
        self.correct = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        self.loss_sum += float(loss.item()) * targets.numel()
        self.count += int(targets.numel())
        self.correct += int((preds == targets).sum().item())
        self.tp += int(((preds == 1) & (targets == 1)).sum().item())
        self.fp += int(((preds == 1) & (targets == 0)).sum().item())
        self.tn += int(((preds == 0) & (targets == 0)).sum().item())
        self.fn += int(((preds == 0) & (targets == 1)).sum().item())

    def summary(self) -> dict:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        accuracy = self.correct / max(self.count, 1)
        return {
            "loss": self.loss_sum / max(self.count, 1),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "count": self.count,
        }


def load_input_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s*\|\s*", engine="python", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    required = ["dt", "open", "close", "high", "low", "vol", "target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input file is missing required columns: {missing}")

    dt_series = df["dt"].astype(str).str.strip()
    separator_mask = dt_series.str.fullmatch(r"[-+]+") | dt_series.str.contains(r"^-{2,}\+", regex=True)
    df = df.loc[~separator_mask].copy()

    df["dt"] = pd.to_datetime(
        df["dt"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        utc=True,
        errors="coerce",
    )
    for col in ["open", "close", "high", "low", "vol", "target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).sort_values("dt").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    feats = np.zeros((n, FEATURE_SIZE), dtype=np.float32)
    if n == 0:
        return feats

    raw_close = df["close"].to_numpy(dtype=np.float64)
    raw_open = df["open"].to_numpy(dtype=np.float64)
    raw_high = df["high"].to_numpy(dtype=np.float64)
    raw_low = df["low"].to_numpy(dtype=np.float64)
    dt = df["dt"]

    for i in range(1, n):
        ref = raw_close[i - 1]
        o = math.log(raw_open[i] / ref) * K_FEATURE_SCALE
        c = math.log(raw_close[i] / ref) * K_FEATURE_SCALE
        h = math.log(raw_high[i] / ref) * K_FEATURE_SCALE
        l = math.log(raw_low[i] / ref) * K_FEATURE_SCALE

        feats[i, 0] = o
        feats[i, 1] = c
        feats[i, 2] = h
        feats[i, 3] = l

        body = c - o
        rng = h - l
        denom = max(rng, 1e-6)
        feats[i, 4] = body
        feats[i, 5] = rng
        feats[i, 12] = (h - max(o, c)) / denom
        feats[i, 13] = (min(o, c) - l) / denom

        start_j = max(1, i - (ROLLING_VOL_LOOKBACK - 1))
        rets = np.log(raw_close[start_j : i + 1] / raw_close[start_j - 1 : i])
        if len(rets) > 1:
            feats[i, 6] = float(np.std(rets) * K_FEATURE_SCALE)
        else:
            feats[i, 6] = 0.0

        start_ret = max(1, i - (ROLLING_RET_LOOKBACK - 1))
        rets2 = np.log(raw_close[start_ret : i + 1] / raw_close[start_ret - 1 : i])
        feats[i, 7] = float(np.sum(rets2) * K_FEATURE_SCALE)

        ts = dt.iloc[i]
        # Mirror the repo's current 15-minute cycle feature.
        epoch_sec = int(ts.timestamp())
        cyc_sec = 15 * 60
        sec_in_cycle = epoch_sec % cyc_sec
        phase = 2.0 * math.pi * (sec_in_cycle / cyc_sec)
        feats[i, 8] = math.sin(phase)
        feats[i, 9] = math.cos(phase)

        sec_of_week = (
            ts.dayofweek * 24 * 60 * 60
            + ts.hour * 60 * 60
            + ts.minute * 60
            + ts.second
        )
        phase_w = 2.0 * math.pi * (sec_of_week / (7 * 24 * 60 * 60))
        feats[i, 10] = math.sin(phase_w)
        feats[i, 11] = math.cos(phase_w)

    return feats


def build_sequences(
    df: pd.DataFrame,
    features: np.ndarray,
    window_size: int,
    prediction_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes = df["close"].to_numpy(dtype=np.float64)
    dts = df["dt"].to_numpy()

    xs: List[np.ndarray] = []
    ys: List[float] = []
    ts_out: List[np.datetime64] = []

    max_start = len(df) - window_size - prediction_horizon + 1
    for start in range(max_start):
        last_idx = start + window_size - 1
        target_idx = last_idx + prediction_horizon
        x = features[start : start + window_size]
        y = 1.0 if closes[target_idx] > closes[last_idx] else 0.0
        xs.append(x)
        ys.append(y)
        ts_out.append(dts[last_idx])

    if not xs:
        raise ValueError("no sequences could be built; check the date range and file length")

    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(ts_out)


def split_train_valid(
    seq_x: np.ndarray,
    seq_y: np.ndarray,
    seq_ts: np.ndarray,
    valid_fraction: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if not (0.0 <= valid_fraction < 0.5):
        raise ValueError("valid_fraction must be in [0.0, 0.5)")

    n = len(seq_y)
    valid_n = max(1, int(n * valid_fraction)) if n >= 10 else 1
    train_n = max(n - valid_n, 1)

    # Time-based split to avoid leakage.
    return (
        (seq_x[:train_n], seq_y[:train_n]),
        (seq_x[train_n:], seq_y[train_n:]),
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    class WindowedDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
        def __init__(self, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
            self.x_arr = x_arr.astype(np.float32, copy=False)
            self.y_arr = y_arr.astype(np.float32, copy=False)

        def __len__(self) -> int:
            return len(self.y_arr)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.from_numpy(self.x_arr[idx]),
                torch.tensor([self.y_arr[idx]], dtype=torch.float32),
            )

    return DataLoader(
        WindowedDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    metrics = WarmupBinaryMetrics()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics.update(logits.detach(), yb.detach(), loss.detach())

    return metrics.summary()


def save_artifacts(
    output_dir: Path,
    model: nn.Module,
    config: TrainConfig,
    history: list[dict],
    train_rows: int,
    valid_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "PyTorch LSTM training output\n"
            f"train_sequences={train_rows}\n"
            f"valid_sequences={valid_rows}\n"
            f"window_size={config.window_size}\n"
            f"prediction_horizon={config.prediction_horizon}\n"
            f"hidden_size={config.hidden_size}\n"
            f"batch_size={config.batch_size}\n"
            f"learning_rate={config.learning_rate}\n"
            "target=BinaryReturn\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch LSTM on input15m.txt")
    parser.add_argument("--input", default="/mnt/data/input15m.txt", help="Path to input15m.txt")
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--prediction-horizon", type=int, default=DEFAULT_PREDICTION_HORIZON)
    parser.add_argument("--c-next-threshold", type=float, default=DEFAULT_C_NEXT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--valid-fraction", type=float, default=DEFAULT_VALID_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--output-dir", default="training_output")
    args = parser.parse_args()

    config = TrainConfig(
        input_path=args.input,
        train_start=args.train_start,
        train_end=args.train_end,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        c_next_threshold=args.c_next_threshold,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )

    set_seed(config.seed)
    device = choose_device(config.device)
    print(f"Using device: {device}")

    df = load_input_file(config.input_path)
    mask = (df["dt"] >= pd.Timestamp(config.train_start, tz="UTC")) & (
        df["dt"] < pd.Timestamp(config.train_end, tz="UTC")
    )
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows remain after applying the training date filter")

    print(f"Rows after date filter: {len(df)}")
    print(f"Date range: {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")

    features = engineer_features(df)
    seq_x, seq_y, _ = build_sequences(
        df=df,
        features=features,
        window_size=config.window_size,
        prediction_horizon=config.prediction_horizon,
    )
    (train_x, train_y), (valid_x, valid_y) = split_train_valid(
        seq_x,
        seq_y,
        np.empty(len(seq_y)),
        config.valid_fraction,
    )

    print(f"Train sequences: {len(train_y)}")
    print(f"Valid sequences: {len(valid_y)}")
    print(f"Positive class ratio (train): {float(train_y.mean()):.4f}")

    train_loader = make_loader(train_x, train_y, config.batch_size, shuffle=True)
    valid_loader = make_loader(valid_x, valid_y, config.batch_size, shuffle=False)

    model = BinaryLSTM(input_size=FEATURE_SIZE, hidden_size=config.hidden_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict] = []
    best_valid_loss = float("inf")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_metrics = run_epoch(model, valid_loader, criterion, device, None)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} "
            f"valid_loss={valid_metrics['loss']:.6f} valid_acc={valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    save_artifacts(output_dir, model, config, history, len(train_y), len(valid_y))
    print(f"Saved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
