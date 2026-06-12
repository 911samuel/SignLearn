# SignLearn — Architecture Diagrams

> Diagrams are written in [Mermaid](https://mermaid.js.org/) — they render natively on GitHub, GitLab, Notion, and in VS Code with the "Markdown Preview Mermaid Support" extension.

---

## 1. System Architecture

```mermaid
flowchart TB
    subgraph BROWSER["Browser  (Next.js 15 · App Router)"]
        direction TB
        CAM["Webcam / Mic"]
        MP["MediaPipe Hands\nin-browser · GPU delegate"]
        STT["Web Speech API\nspeech → text"]

        subgraph UI["React Components"]
            SP["SignerPanel"]
            HP["HearingPanel"]
            CP["CaptionsPanel"]
            CM["ConfidenceMeter"]
            LO["LandmarkOverlay canvas"]
        end
    end

    subgraph NET["WebSocket  (Socket.IO)"]
        direction LR
        FE["frame  {landmarks: 126 floats, t}"]
        PR["prediction  {label, confidence, ready}"]
    end

    subgraph BACKEND["Backend  (Flask + Socket.IO · :5001)"]
        direction TB
        SH["socket_handlers.py"]
        FB["FrameBuffer\n30-frame sliding window"]
        NM["normalize_frame\nwrist-centred · unit-scaled"]
        ML["ModelLoader  (ONNX Runtime)"]
        PS["PredictionSmoother\nEMA α=0.6 · conf ≥ 0.75"]

        subgraph API["REST API"]
            H["/health"]
            MET["/metrics  (Prometheus)"]
            ADM["/admin/reload  hot-swap"]
            TRX["/transcript"]
        end

        DB[("SQLite\nsignlearn.sqlite")]
    end

    subgraph ONNX["TCN ONNX Model"]
        TCN["Dilated TCN\n126 K params · p95 = 0.23 ms\n4 889 fps on CPU"]
    end

    CAM --> MP
    MP -->|"126 floats / frame"| SP
    SP --> FE
    FE --> SH
    SH --> FB --> NM --> ML
    ML <--> TCN
    ML --> PS --> PR
    PR --> CM
    PR --> CP
    STT --> HP
    CP --> HP
    SH --> DB
    DB --> TRX
```

---

## 2. Real-Time Data Flow (Sequence)

```mermaid
sequenceDiagram
    actor Signer
    participant Browser as Browser (MediaPipe)
    participant WS as WebSocket
    participant Buf as FrameBuffer (30 frames)
    participant Model as TCN ONNX
    participant Smoother as PredictionSmoother
    participant Hearing as Hearing User

    Signer->>Browser: Signs in front of camera
    loop Every video frame (~30 fps)
        Browser->>Browser: MediaPipe extracts 21 landmarks/hand
        Browser->>WS: emit("frame", {landmarks: [126], t})
        WS->>Buf: push(frame)
        alt Buffer not full yet
            Buf-->>WS: emit("prediction", {ready: false})
        else Buffer full (30 frames)
            Buf->>Model: run_inference(window)
            Model-->>Buf: softmax[36]
            Buf->>Smoother: update(probs)
            alt confidence >= 0.75
                Smoother-->>WS: emit("prediction", {label, confidence, ready: true})
                WS-->>Browser: prediction event
                Browser-->>Signer: ConfidenceMeter lights up
                Browser-->>Hearing: Caption appears
            else confidence < 0.75
                Smoother-->>WS: emit("prediction", {label: null, ready: false})
            end
        end
    end
```

---

## 3. ML Training Pipeline

```mermaid
flowchart LR
    subgraph DATA["Data Layer"]
        KAG["Kaggle ASL\nAlphabet + Digits"]
        RAW["data/raw/\n.jpg images"]
        EXT["extract_landmarks.py\nMediaPipe → .npy"]
        PRO["data/processed/\ntrain / val / test\n66 666 sequences\n(30 × 126) float32"]
        AUG["augment.py\nrotate · scale · noise\nrotate3d · speed_warp"]
    end

    subgraph FEAT["Feature Engineering"]
        FM["features.py\nraw (126)\nraw+velocity (252)\nengineered (398)"]
    end

    subgraph TRAIN["Training"]
        CFG["TrainConfig\narch · feature_mode · lr · dropout"]
        TR["train_model.py\nAdam · class-weighted loss\nReduceLROnPlateau\nEarlyStopping"]
        TB["TensorBoard\nartifacts/runs/*/logs/"]
        CKP["best.keras\nartifacts/runs/*/checkpoints/"]
    end

    subgraph EVAL["Evaluation"]
        EV["evaluate_model.py\ntest acc · macro F1\nconfusion matrix"]
        SW["sweep.py\n16 runs: 4 arch × 2 feat × 2 lr"]
        RPT["model_comparison.md"]
    end

    subgraph SERVE["Serving"]
        EXP["export_onnx.py\ntf2onnx · opset 17\nparity check max Δ < 1e-4"]
        PRF["profile_inference.py\np95 = 0.23 ms"]
        ONX["tcn_best.onnx\nartifacts/checkpoints/"]
        API["Flask API\n:5001"]
    end

    KAG --> RAW --> EXT --> PRO
    PRO --> AUG --> FM --> CFG --> TR
    TR --> TB
    TR --> CKP --> EV
    EV --> SW --> RPT
    CKP --> EXP --> ONX --> PRF
    ONX --> API
```

---

## 4. Component Hierarchy (Frontend)

```mermaid
flowchart TD
    LAY["layout.tsx\nRootLayout · ThemeToggle · ToastProvider"]

    LAY --> HOME["page.tsx  (Landing)\nLandingCTA · Create / Join room"]
    LAY --> PRAC["practice/page.tsx\nSolo practice · no room needed"]
    LAY --> LEARN["learn/page.tsx\nSign dictionary"]
    LAY --> ROOM["r/[roomId]/page.tsx\nRoomInner · ConversationLog"]
    LAY --> JOIN["r/[roomId]/join/page.tsx\nRole selection · name entry"]

    ROOM --> SV["SignerView\ncamera · signs → captions"]
    ROOM --> HV["HearingView\nmic · speech → captions"]

    SV --> VID["video element\n+ LandmarkOverlay canvas"]
    SV --> CM["ConfidenceMeter\nglow on commit"]
    SV --> RV1["RemoteVideo (peer)"]
    SV --> CP1["CaptionsPanel (speech)"]

    HV --> RV2["RemoteVideo (signer)"]
    HV --> RV3["RemoteVideo (self, muted)"]
    HV --> CP2["CaptionsPanel (sign)"]
    HV --> PTT["Push-to-talk button\n+ Space-bar hold"]

    PRAC --> VID2["video + LandmarkOverlay"]
    PRAC --> CM2["ConfidenceMeter"]
    PRAC --> HIST["History chips\npop animation"]
```

---

## 5. Model Architecture — TCN

```mermaid
flowchart TD
    IN["Input  (30 × 126)\n30 frames · 126 landmark features"]
    PROJ["Conv1D k=1\nProjection → 64 filters"]

    subgraph RES1["Residual Block  dilation=1"]
        C1A["Conv1D k=3 · d=1 · 64"]
        LN1A["LayerNorm → ReLU → Dropout 0.4"]
        C1B["Conv1D k=3 · d=1 · 64"]
        LN1B["LayerNorm → ReLU → Dropout 0.4"]
    end

    subgraph RES2["Residual Block  dilation=2"]
        C2A["Conv1D k=3 · d=2 · 64"]
        LN2A["LayerNorm → ReLU → Dropout 0.4"]
        C2B["Conv1D k=3 · d=2 · 64"]
        LN2B["LayerNorm → ReLU → Dropout 0.4"]
    end

    subgraph RES4["Residual Block  dilation=4"]
        C4A["Conv1D k=3 · d=4 · 64"]
        LN4B["LayerNorm → ReLU → Dropout 0.4"]
    end

    subgraph RES8["Residual Block  dilation=8"]
        C8A["Conv1D k=3 · d=8 · 64"]
        LN8B["LayerNorm → ReLU → Dropout 0.4"]
    end

    GAP["GlobalAveragePooling1D"]
    D1["Dense 128 · ReLU · Dropout 0.4"]
    OUT["Dense 36 · Softmax\n36 classes: a-z + zero-nine"]

    IN --> PROJ --> RES1 --> RES2 --> RES4 --> RES8 --> GAP --> D1 --> OUT

    STATS["126 372 params · 0.23 ms p95 · 97.84% test acc"]
    OUT -. results .- STATS
```

---

## Quick Reference

| Layer | Technology | Key numbers |
|---|---|---|
| Frontend | Next.js 15 · React · Socket.IO client | Hot reload, App Router |
| Hand tracking | MediaPipe Hands (in-browser) | 21 landmarks × 2 hands × 3 coords = 126 floats/frame |
| Transport | WebSocket (Socket.IO) | `frame` → server, `prediction` ← server |
| Frame buffer | Python (server) | 30 frames sliding window, stride = 1 |
| Model | TCN · ONNX Runtime | 126 K params · p95 = 0.23 ms · 4 889 fps |
| Accuracy | 36-class test set | 97.84% accuracy · 95.5% macro F1 |
| Backend | Flask + Flask-SocketIO | Port 5001 · threading mode |
| Storage | SQLite | Transcript · feedback · corrections |
