"""
Produce publication-quality figures for the SignLearn thesis (Chapters 4-5).

Reads existing artefacts (no retraining). Writes 300dpi PNGs into
artifacts/thesis/figures/. Each figure is plain matplotlib — no seaborn or
exotic dependencies — so it renders identically inside any LaTeX/Word toolchain.

Usage:
    python backend/scripts/build_thesis_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).parent.parent.parent
RUNS = REPO / "artifacts" / "runs"
REPORTS = REPO / "artifacts" / "reports"
OUT = REPO / "artifacts" / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PROD = RUNS / "phase3-raw-balanced__arch=tcn_fm=raw_lr=0.0005_do=0.4"
WORD = RUNS / "word-aslc-tcn-78cls-v1"
WORD_N_CLASSES = 78

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
# Diagrams (manually drawn block diagrams via matplotlib patches)
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, color="#E8F0FE", edge="#1A73E8", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        linewidth=1.2, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#5F6368", lw=1.4))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label,
                ha="center", fontsize=8, color="#5F6368")


def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Figure: SignLearn End-to-End System Architecture", pad=10)

    _box(ax, 0.2, 2.0, 1.8, 1.0, "Webcam\n(browser)", color="#FFF4E5", edge="#E37400")
    _box(ax, 2.4, 2.0, 2.0, 1.0, "MediaPipe Hands\n(in-browser)", color="#FFF4E5", edge="#E37400")
    _box(ax, 4.8, 3.2, 1.8, 0.9, "WebSocket\n(landmarks)", color="#F1F3F4", edge="#5F6368")
    _box(ax, 4.8, 0.9, 1.8, 0.9, "Speech-to-Text\n(Web Speech API)", color="#FFF4E5", edge="#E37400")
    _box(ax, 7.0, 2.0, 2.0, 1.0, "FrameBuffer →\nONNX (TCN)", color="#E6F4EA", edge="#137333")
    _box(ax, 9.2, 3.2, 1.6, 0.9, "Smoother\n(EMA + gate)", color="#E6F4EA", edge="#137333")
    _box(ax, 9.2, 0.9, 1.6, 0.9, "SQLite\ntranscript", color="#FCE8E6", edge="#C5221F")

    _arrow(ax, 2.0, 2.5, 2.4, 2.5)
    _arrow(ax, 4.4, 2.5, 4.8, 3.0)
    _arrow(ax, 6.6, 3.6, 7.0, 3.0)
    _arrow(ax, 9.0, 2.5, 9.2, 3.6)
    _arrow(ax, 9.0, 2.5, 9.2, 1.3)
    _arrow(ax, 6.6, 1.3, 7.0, 2.0)

    ax.text(5.5, 4.6, "Browser (client)", fontsize=11, ha="center",
            style="italic", color="#5F6368")
    ax.text(8.5, 4.6, "Flask + SocketIO server", fontsize=11, ha="center",
            style="italic", color="#5F6368")
    ax.axvline(6.8, color="#DADCE0", linestyle="--", lw=1)
    save(fig, "fig_system_architecture.png")


def fig_data_flow():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Figure: Data Flow Diagram (Level 1)", pad=10)

    boxes = [
        (0.3, 1.5, 1.6, 1.0, "Signer\n(external)"),
        (2.3, 1.5, 1.6, 1.0, "1.0\nCapture\nLandmarks"),
        (4.3, 1.5, 1.6, 1.0, "2.0\nNormalize &\nBuffer"),
        (6.3, 1.5, 1.6, 1.0, "3.0\nClassify\n(ONNX)"),
        (8.3, 1.5, 1.6, 1.0, "4.0\nSmooth &\nEmit"),
    ]
    for (x, y, w, h, t) in boxes:
        color = "#FCE8E6" if "external" in t else "#E8F0FE"
        edge = "#C5221F" if "external" in t else "#1A73E8"
        _box(ax, x, y, w, h, t, color=color, edge=edge)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        _arrow(ax, x1, 2.0, x2, 2.0)

    _box(ax, 2.3, 0.1, 1.6, 0.7, "D1: Raw frames", color="#FFFFFF", edge="#5F6368")
    _box(ax, 4.3, 0.1, 1.6, 0.7, "D2: Norm sequence", color="#FFFFFF", edge="#5F6368")
    _box(ax, 6.3, 0.1, 1.6, 0.7, "D3: Softmax probs", color="#FFFFFF", edge="#5F6368")
    _box(ax, 8.3, 0.1, 1.6, 0.7, "D4: Transcript", color="#FFFFFF", edge="#5F6368")

    for (x, y, w, h, _) in boxes[1:]:
        ax.annotate("", xy=(x + w / 2, 0.8), xytext=(x + w / 2, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#9AA0A6", lw=0.8))

    _box(ax, 0.3, 3.0, 10.4, 0.6, "Hearing User: receives 'prediction' WebSocket events + transcript view",
         color="#F1F3F4", edge="#5F6368", fontsize=9)
    save(fig, "fig_data_flow.png")


def fig_sequence_diagram():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Figure: Sequence Diagram — One Prediction Round-Trip", pad=10)

    actors = [
        (1.0, "Browser\n(MediaPipe)"),
        (3.5, "WebSocket\nServer"),
        (6.0, "FrameBuffer"),
        (8.5, "ONNX\nModel"),
    ]
    for x, name in actors:
        _box(ax, x - 0.7, 9.0, 1.4, 0.8, name, color="#E8F0FE", edge="#1A73E8")
        ax.plot([x, x], [9.0, 0.5], color="#DADCE0", linestyle="--", lw=1)

    msgs = [
        (1.0, 3.5, 8.4, 'emit("frame", {landmarks, t})'),
        (3.5, 6.0, 7.6, "push(landmarks)"),
        (6.0, 6.0, 6.8, "if len == 30: normalize_sequence()"),
        (6.0, 8.5, 6.0, "run_inference_probs(seq)"),
        (8.5, 6.0, 5.2, "softmax (36,)"),
        (6.0, 3.5, 4.4, "smoother.update(probs)"),
        (3.5, 1.0, 3.6, 'emit("prediction", {label, conf})'),
        (1.0, 3.5, 2.8, "render confidence meter"),
        (3.5, 1.0, 2.0, "log transcript entry"),
    ]
    for x1, x2, y, label in msgs:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#137333", lw=1.4))
        ax.text((x1 + x2) / 2, y + 0.15, label, ha="center", fontsize=9)

    save(fig, "fig_sequence_diagram.png")


def fig_class_diagram():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Figure: Class Diagram — Backend Core", pad=10)

    klasses = [
        (0.3, 4.0, 2.6, 2.5, "FrameBuffer",
         "+ push(landmarks)\n+ is_ready(): bool\n+ get_sequence(): ndarray\n+ reset()"),
        (3.5, 4.0, 2.6, 2.5, "ModelHolder",
         "+ load(path)\n+ reload(path)\n+ run_inference_probs(x)\n+ get_class_names()"),
        (6.7, 4.0, 2.6, 2.5, "PredictionSmoother",
         "+ update(probs): label, conf\n- ema_alpha: 0.6\n- conf_threshold: 0.75\n- repeat_cooldown: 15"),
        (1.9, 0.8, 2.6, 2.5, "ErrorHandler",
         "+ register_error_handlers(app)\n+ SignLearnError\n  ↳ LandmarkValidationError (422)\n  ↳ ModelNotReadyError (503)"),
        (5.1, 0.8, 2.6, 2.5, "Metrics",
         "+ record_prediction()\n+ record_no_hand_frame()\n+ prometheus_text(): str"),
    ]
    for x, y, w, h, name, methods in klasses:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, lw=1.2,
                                        edgecolor="#1A73E8", facecolor="#E8F0FE"))
        ax.plot([x, x + w], [y + h - 0.5, y + h - 0.5], color="#1A73E8", lw=1.2)
        ax.text(x + w / 2, y + h - 0.25, name, ha="center", va="center",
                fontweight="bold", fontsize=11)
        ax.text(x + 0.15, y + h - 0.7, methods, ha="left", va="top", fontsize=8.5)

    save(fig, "fig_class_diagram.png")


def fig_deployment():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Figure: Deployment View (Local Demo Configuration)", pad=10)

    _box(ax, 0.5, 2.0, 3.0, 2.2, "Client Browser\n\nNext.js 15 App\nMediaPipe Hands (WASM)\nWeb Speech API",
         color="#FFF4E5", edge="#E37400", fontsize=10)
    _box(ax, 4.5, 2.0, 3.0, 2.2, "Flask + SocketIO\n(threading mode)\n\nPython 3.11\nport 5001",
         color="#E6F4EA", edge="#137333", fontsize=10)
    _box(ax, 8.0, 3.2, 1.8, 1.0, "ONNX Runtime\nCPU", color="#F1F3F4", edge="#5F6368", fontsize=9)
    _box(ax, 8.0, 2.0, 1.8, 1.0, "SQLite\nfile", color="#F1F3F4", edge="#5F6368", fontsize=9)

    _arrow(ax, 3.5, 3.1, 4.5, 3.1, "WebSocket")
    _arrow(ax, 7.5, 3.6, 8.0, 3.7)
    _arrow(ax, 7.5, 2.7, 8.0, 2.5)

    ax.text(5, 0.7, "Localhost (127.0.0.1) — single MacBook hosts both client tab and server",
            ha="center", fontsize=10, style="italic", color="#5F6368")
    save(fig, "fig_deployment.png")


# ---------------------------------------------------------------------------
# Data-driven figures
# ---------------------------------------------------------------------------

def fig_training_curves():
    hist = json.load(open(PROD / "reports" / "history.json"))
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, hist["accuracy"], label="Train", color="#1A73E8", lw=1.8)
    axes[0].plot(epochs, hist["val_accuracy"], label="Validation", color="#E37400", lw=1.8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[0].set_ylim(0, 1.02)

    axes[1].plot(epochs, hist["loss"], label="Train", color="#1A73E8", lw=1.8)
    axes[1].plot(epochs, hist["val_loss"], label="Validation", color="#E37400", lw=1.8)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.suptitle("Figure: Letter/Digit Model (TCN, raw, lr=5e-4) — Training History",
                 fontsize=12, y=1.02)
    save(fig, "fig_training_curves.png")


def fig_per_class_accuracy():
    metrics = json.load(open(PROD / "reports" / "metrics.json"))
    per_class = metrics["per_class"]
    classes = sorted(per_class.keys(), key=lambda c: per_class[c]["recall"])
    recalls = [per_class[c]["recall"] for c in classes]
    supports = [per_class[c]["support"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#C5221F" if r < 0.6 else "#E37400" if r < 0.85 else "#137333"
              for r in recalls]
    ax.barh(classes, recalls, color=colors, edgecolor="#202124", lw=0.4)
    ax.set_xlabel("Recall (per-class)")
    ax.set_title("Figure: Per-Class Recall on Test Set (TCN, 36 classes)")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.85, color="#5F6368", linestyle=":", lw=1)
    for c, r, s in zip(classes, recalls, supports):
        ax.text(min(r + 0.01, 0.95), classes.index(c),
                f"{r*100:.0f}% (n={s})", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    legend = [
        mpatches.Patch(color="#137333", label="≥ 85% recall"),
        mpatches.Patch(color="#E37400", label="60–85%"),
        mpatches.Patch(color="#C5221F", label="< 60% (linguistic ambiguity)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    save(fig, "fig_per_class_accuracy.png")


def fig_architecture_comparison():
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.name.startswith("phase3-raw-balanced__"):
            continue
        m_path = d / "reports" / "metrics.json"
        if not m_path.exists():
            continue
        m = json.load(open(m_path))
        # Parse arch/feature mode/lr from name
        parts = d.name.split("__", 1)[1].split("_")
        cfg = {}
        for tok in parts:
            if "=" in tok:
                k, v = tok.split("=", 1)
                cfg[k] = v
        rows.append((cfg.get("arch", "?"), cfg.get("fm", "?"),
                     cfg.get("lr", "?"), m["accuracy"]))
    rows.sort(key=lambda r: -r[3])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [f"{r[0]}\n{r[1]}, lr={r[2]}" for r in rows]
    accs = [r[3] for r in rows]
    bars = ax.bar(range(len(labels)), accs,
                  color=["#137333" if a >= 0.97 else "#1A73E8" if a >= 0.95 else "#E37400" if a >= 0.85 else "#C5221F"
                         for a in accs],
                  edgecolor="#202124", lw=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Figure: Phase-3 Architecture × Feature × LR Sweep Results")
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.95, color="#5F6368", linestyle=":", lw=1)
    ax.text(len(labels) - 0.5, 0.955, "95% target", fontsize=8, ha="right", color="#5F6368")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.01, f"{a*100:.1f}%", ha="center", fontsize=8)
    save(fig, "fig_architecture_comparison.png")


def fig_word_model_top5():
    test = json.load(open(WORD / "reports" / "test_metrics.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = [f"Random\n(1/{WORD_N_CLASSES})", "Top-1", "Top-5",
               f"Random\n(5/{WORD_N_CLASSES})"]
    values = [1/WORD_N_CLASSES, test["accuracy"], test["top5_acc"], 5/WORD_N_CLASSES]
    colors = ["#9AA0A6", "#1A73E8", "#137333", "#9AA0A6"]
    ax.bar(metrics, values, color=colors, edgecolor="#202124", lw=0.4)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Figure: Production Word Model — Top-1 vs Top-5 vs Random "
                 f"({WORD_N_CLASSES} ASL Citizen classes)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_word_model_top5.png")


def fig_latency():
    lat = json.load(open(REPORTS / "phase5_latency.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    pts = ["mean", "p50", "p95", "p99"]
    vals = [lat.get(f"{k}_ms", 0) for k in pts]
    bars = ax.bar(pts, vals, color="#1A73E8", edgecolor="#202124", lw=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f} ms",
                ha="center", fontsize=10)
    ax.axhline(2000, color="#137333", linestyle="--", lw=1, label="2 s target")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Figure: End-to-End WebSocket Round-Trip Latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_latency.png")


def fig_confusion_matrix_copy():
    """Copy the existing confusion matrix so all figs live in one place."""
    src = PROD / "reports" / "confusion_matrix.png"
    if src.exists():
        dst = OUT / "fig_confusion_matrix.png"
        shutil.copy2(src, dst)
        print(f"  copied {dst}")


if __name__ == "__main__":
    print("Diagrams:")
    fig_system_architecture()
    fig_data_flow()
    fig_sequence_diagram()
    fig_class_diagram()
    fig_deployment()
    print("Data-driven:")
    fig_training_curves()
    fig_per_class_accuracy()
    fig_architecture_comparison()
    fig_word_model_top5()
    fig_latency()
    fig_confusion_matrix_copy()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} figures in {OUT}")
