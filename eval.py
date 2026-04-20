"""
평가 지표 계산 및 시각화.

주요 함수:
  full_report(model, dataset)   → 콘솔 출력 + 그래프 저장
  plot_zone_map(model, snap)    → 단일 스냅샷 존 지도
  plot_roc_curve(model, dataset) → ROC 커브
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import GraphDataset, GraphSnapshot
from model import ZoneRiskGNN

OUT_DIR  = Path(__file__).parent.parent / "outputs"
DATA_DIR = Path(__file__).parent.parent / "data"


def _to_np(t: torch.Tensor) -> np.ndarray:
    """torch → numpy. numpy 브릿지가 없는 환경에서도 동작."""
    return np.array(t.detach().tolist(), dtype=np.float32)


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    model: ZoneRiskGNN,
    dataset: GraphDataset,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    all_prob:  List[np.ndarray] = []
    all_label: List[np.ndarray] = []

    for snap in dataset:
        all_prob.append(_to_np(model(snap)))
        all_label.append(_to_np(snap.y))

    prob  = np.concatenate(all_prob)
    label = np.concatenate(all_label).astype(int)
    pred  = (prob > threshold).astype(int)

    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())

    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    auc  = _roc_auc(label, prob)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.linspace(0, 1, 101)[::-1]
    tprs, fprs = [], []
    for thr in thresholds:
        p  = (scores > thr).astype(int)
        tp = int(((p == 1) & (labels == 1)).sum())
        fp = int(((p == 1) & (labels == 0)).sum())
        fn = int(((p == 0) & (labels == 1)).sum())
        tn = int(((p == 0) & (labels == 0)).sum())
        tprs.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))
    return float(np.trapz(tprs, fprs) )


# ── 존별 분석 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def zone_analysis(
    model: ZoneRiskGNN,
    dataset: GraphDataset,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """존별 정확도, 평균 예측 확률, 양성 비율을 DataFrame으로 반환."""
    model.eval()
    n_zones = dataset.n_zones
    records: Dict[int, List] = {z: [] for z in range(n_zones)}

    for snap in dataset:
        prob  = _to_np(model(snap))
        label = _to_np(snap.y)
        for z in range(n_zones):
            records[z].append((float(prob[z]), float(label[z])))

    rows = []
    for z, data in records.items():
        probs  = np.array([d[0] for d in data])
        labels = np.array([d[1] for d in data])
        preds  = (probs > threshold).astype(int)
        rows.append(dict(
            zone_id   = z,
            accuracy  = round(float((preds == labels).mean()), 3),
            avg_prob  = round(float(probs.mean()), 3),
            pos_ratio = round(float(labels.mean()), 3),
            zone_flow = round(float(dataset.zone_flow[z].item()), 3),
        ))
    return pd.DataFrame(rows).set_index("zone_id")


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_zone_map(
    model: ZoneRiskGNN,
    snap: GraphSnapshot,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    title: str = "Zone Risk Prediction",
    save_path: Path | None = None,
) -> None:
    model.eval()
    with torch.no_grad():
        probs  = _to_np(model(snap))
    labels = _to_np(snap.y)
    preds  = (probs > 0.5).astype(int)

    grid = int(round(len(probs) ** 0.5))
    cell = 100 / grid

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title}  [{snap.date}]", fontsize=13)

    for ax, arr, sub in zip(axes, [labels, preds], ["Ground Truth", "GNN Prediction"]):
        ax.set_title(sub, fontsize=11)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel("x"); ax.set_ylabel("y")

        for z in range(len(probs)):
            col_i = z % grid
            row_i = z // grid
            color = "#e74c3c" if arr[z] == 1 else "#2ecc71"
            alpha = 0.2 + 0.55 * probs[z] if sub == "GNN Prediction" else 0.3
            rect  = patches.Rectangle(
                (col_i * cell, row_i * cell), cell, cell,
                linewidth=1, edgecolor="#888", facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)
            ax.text(
                col_i * cell + cell / 2, row_i * cell + cell / 2,
                f"{probs[z]:.2f}", ha="center", va="center", fontsize=8,
            )

        for _, row in edges_df.iterrows():
            u, v = int(row["src"]), int(row["dst"])
            xu, yu = nodes_df.loc[u, "x"], nodes_df.loc[u, "y"]
            xv, yv = nodes_df.loc[v, "x"], nodes_df.loc[v, "y"]
            ax.plot([xu, xv], [yu, yv], color="#555",
                    lw=0.3 + 1.8 * row["avg_flow"], alpha=0.25, zorder=1)

        avg_s = (_to_np(snap.x[:, 0]) + _to_np(snap.x[:, 1])) / 2
        sc = ax.scatter(
            nodes_df["x"], nodes_df["y"],
            c=avg_s, cmap="YlOrRd", vmin=0, vmax=1,
            s=55, zorder=3, edgecolors="k", linewidths=0.4,
        )
        if sub == "Ground Truth":
            plt.colorbar(sc, ax=ax, label="avg score (A+B)/2", shrink=0.75)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.show()


def plot_roc_curve(
    model: ZoneRiskGNN,
    dataset: GraphDataset,
    save_path: Path | None = None,
) -> None:
    all_prob, all_label = [], []
    model.eval()
    with torch.no_grad():
        for snap in dataset:
            all_prob.append(_to_np(model(snap)))
            all_label.append(_to_np(snap.y))

    prob  = np.concatenate(all_prob)
    label = np.concatenate(all_label).astype(int)

    thresholds = np.linspace(0, 1, 201)[::-1]
    tprs, fprs = [], []
    for thr in thresholds:
        p  = (prob > thr).astype(int)
        tp = int(((p == 1) & (label == 1)).sum())
        fp = int(((p == 1) & (label == 0)).sum())
        fn = int(((p == 0) & (label == 1)).sum())
        tn = int(((p == 0) & (label == 0)).sum())
        tprs.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))

    auc = float(np.trapz(tprs, fprs) )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, color="#2980b9", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Zone Risk Classifier")
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.show()


# ── 통합 리포트 ────────────────────────────────────────────────────────────────

def full_report(
    model: ZoneRiskGNN,
    val_dataset: GraphDataset,
    threshold: float = 0.5,
    save_dir: Path = OUT_DIR,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(model, val_dataset, threshold)
    print("\n── Validation Metrics ─────────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<12}: {v:.4f}")
        else:
            print(f"  {k:<12}: {v}")

    df_zone = zone_analysis(model, val_dataset, threshold)
    print("\n── Per-Zone Analysis ──────────────────────────────")
    print(df_zone.to_string())
    df_zone.to_csv(save_dir / "zone_analysis.csv")

    nodes_df = pd.read_csv(DATA_DIR / "nodes.csv").set_index("node_id")
    edges_df = pd.read_csv(DATA_DIR / "edges.csv")
    last_snap = val_dataset[-1]
    plot_zone_map(
        model, last_snap, nodes_df, edges_df,
        title="Zone Risk Map (last validation snapshot)",
        save_path=save_dir / "zone_map.png",
    )
    plot_roc_curve(model, val_dataset, save_path=save_dir / "roc_curve.png")
