# PMUT Frequency Response Prediction — Experiment Flowchart

```mermaid
flowchart TD

    %% ─── ROOT ───────────────────────────────────────────────────────────────
    DATA["📁  PMUT Dataset
    16 devices · 1 000 processed frequency points each
    Frequency Response Data I.xlsx
    Frequency Response Data II.xlsx"]

    PREP["⚙️  Preprocessing
    Merge averaged Excel sheets · Convert Hz → MHz
    Amplitude scaling in data loader
    ▶  data/processed/pmuts_all.csv"]

    DATA --> PREP

    %% ─── STRATEGY HEADERS ──────────────────────────────────────────────────
    PREP --> S06 & S07

    S06["📈  Progressive Training"]

    S07["🔄  Regime-Aware Training"]

    %% ─── PROGRESSIVE LOOP ────────────────────────────────────────────────────
    subgraph PROG ["Progressive loop  ·  model retrained with cumulative PMUTs (target N = 2 → 16)"]
        direction TB

        PL1["① Initialise
        Train model on { P₁ }"]

        PL2["② Predict P(N+1)
        Input  : frequency vector [ f₁, f₂, … f₁₀₀₀ ] for N_PMUT
        Output : amplitude vector [ a₁, a₂, … a₁₀₀₀ ]
        — Vector / Multi-Output —"]

        PL3["③ Evaluate on Full Spectrum
        MAE  ·  R²"]

        PL4["④ Expand training set
        Append P(N+1) to training data
        Re-train same model on
        { P₁, P₂, … P(N+1) }"]

        DONE06(["✓  15 progressive targets
        (target N = 2 → 16)"])

        PL1 --> PL2
        PL2 --> PL3
        PL3 --> PL4
        PL4 -->|"N+1 < 16   ↺   keep model, grow set"| PL2
        PL4 --> DONE06
    end

    %% ─── REGIME BLOCKS — horizontal ─────────────────────────────────────────
    subgraph REGS ["Regime-Aware  ·  fresh model instance at every regime boundary"]
        direction LR

        R1["Regime 1
        PMUTs 1 – 2
        ─────────────
        Train [1] → Predict 2"]

        R2["Regime 2
        PMUTs 3 – 11
        ─────────────
        Train [3] → Predict 4
        Train [3,4] → Predict 5
               ⋮
        Train [3..10] → Predict 11"]

        R3["Regime 3
        PMUTs 12 – 16
        ─────────────
        Train [12] → Predict 13
               ⋮
        Train [12..15] → Predict 16"]
    end

    %% ─── CONNECT STRATEGIES TO REGIME BLOCKS ────────────────────────────────
    S06 --> PROG
    S07 --> REGS

    %% ─── SHARED VECTOR PREDICTION BLOCK ─────────────────────────────────────
    DONE06 --> VPRED
    R1 & R2 & R3 --> VPRED

    VPRED["🔷  Vector Prediction
    Input  : frequency vector [ f₁, f₂, … f₁₀₀₀ ] for N_PMUT
    Output : amplitude vector [ a₁, a₂, … a₁₀₀₀ ]
    Evaluated on Full Spectrum  ·  MAE · R²"]

    %% ─── SHARED MODEL BLOCKS ─────────────────────────────────────────────────
    VPRED --> RF["🌲  RF"]
    VPRED --> GB["📈  GB"]
    VPRED --> MLP["🧠  MLP"]
    VPRED --> PINN["⚙️  PINN"]
    VPRED --> MDOFPINN["⚙️  MDOF-PINN"]
    VPRED --> PINNLOSS["⚙️  PINN-Informed-Loss"]
    VPRED --> HYBRID["🔀  RF-MDOF-PINN"]

    %% ─── OUTPUT ──────────────────────────────────────────────────────────────
    RF & GB & MLP & PINN & MDOFPINN & PINNLOSS & HYBRID --> OUT["📄  results/metrics/
    progressive_metrics.csv
    regime_metrics.csv
    supplementary/explorer/
    figures/main/"]

    %% ─── STYLES ─────────────────────────────────────────────────────────────
    classDef data  fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef prep  fill:#E8EAF6,stroke:#3949AB,color:#1A237E
    classDef hdr06 fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef hdr07 fill:#FFF3E0,stroke:#E65100,color:#BF360C
    classDef loop  fill:#F1F8E9,stroke:#558B2F,color:#1B5E20
    classDef done  fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef reg   fill:#FFF8E1,stroke:#F9A825,color:#E65100
    classDef vpred fill:#EDE7F6,stroke:#4527A0,color:#311B92
    classDef rf    fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef gb    fill:#FBE9E7,stroke:#BF360C,color:#BF360C
    classDef mlp   fill:#EDE7F6,stroke:#4527A0,color:#311B92
    classDef out   fill:#FFF9C4,stroke:#F57F17,color:#E65100

    class DATA data
    class PREP prep
    class S06 hdr06
    class S07 hdr07
    class PL1,PL2,PL3,PL4 loop
    class DONE06 done
    class R1,R2,R3 reg
    class VPRED vpred
    class RF rf
    class GB gb
    class MLP mlp
    class OUT out
```

---

## What the diagram shows

```
Data
 └── Preprocessing
       ├── Progressive Training
       │     └── LOOP ↺ (model retained, training set grows each step)
       │           ①  Train on {P1}
       │           ②  Predict P(N+1)
       │           ③  Evaluate  MAE · R²
       │           ④  Append P(N+1), re-train  →  back to ② if N+1 < 16
       │           Done (N=16) ──────────────────────────────┐
       │                                                      │
       └── Regime-Aware Training                             │
             ├─ [Regime 1: PMUTs 1–2 ] ─────────────────────┤
             ├─ [Regime 2: PMUTs 3–11] ─────────────────────┤
             └─ [Regime 3: PMUTs 12–16] ────────────────────┤
                                                             ▼
                          🔷  Vector Prediction
                          Input : freq vector [f₁…f₁₀₀₀] for N_PMUT
                          Output: amplitude vector [a₁…a₁₀₀₀]
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                       🌲 RF         📈 GB         🧠 MLP
                                        │
                                       OUT
```

## Colour key

| Colour | Meaning |
|--------|---------|
| 🔵 Blue | Dataset |
| 🟣 Indigo | Preprocessing |
| 🟢 Green | Progressive Training header + loop nodes |
| 🟠 Orange | Regime-Aware Training header + regime blocks |
| 💜 Purple | Shared Vector Prediction block |
| 🔵 Light blue | RF model |
| 🔴 Coral | GB model |
| 💜 Lavender | MLP model |
| 🟡 Yellow | Output files |
