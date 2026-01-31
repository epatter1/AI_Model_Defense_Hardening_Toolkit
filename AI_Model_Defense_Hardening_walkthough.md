# 🛡️ **Week 3–4: AI Defense Toolkit – Complete Walkthrough**

## 📋 Overview

Hands‑on implementation of **model hardening techniques** including adversarial training, differential privacy, robustness evaluation, and multi‑cloud defense comparisons.

**Weeks 3–4 Goals:**  
- ✅ Implement adversarial training pipelines  
- ✅ Add differential privacy (DP‑SGD) examples  
- ✅ Build robustness evaluation dashboard  
- ✅ Compare defenses across Azure, AWS, and GCP  
- ✅ Create Model Hardening Playbook  

**Time:** 8–10 hours  
**Prerequisites:** Completed Module 1 baseline model + attacks, Python 3.9+, PyTorch/TensorFlow basics  

**Next Module:** Model Monitoring + AI Guardrails (Module 3)

---

# 🚀 **Step-by-Step Implementation**

---

## **Step 1: Project Setup (15 minutes)**

```bash
# Create project structure
mkdir module2-defense-toolkit
cd module2-defense-toolkit

# Create directory structure
mkdir -p adversarial-training
mkdir -p differential-privacy
mkdir -p robustness-dashboard
mkdir -p cloud-comparison
mkdir -p diagrams
mkdir -p docs/case_studies
mkdir -p models
mkdir -p data

# Initialize Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
# ML/AI Libraries
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Adversarial ML
foolbox>=3.3.3
art>=1.14.0
cleverhans>=4.0.0

# Differential Privacy
opacus>=1.4.0
tensorflow-privacy>=0.9.0

# Dashboard
streamlit>=1.29.0
plotly>=5.17.0

# Utilities
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
EOF

pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "import torch, art, opacus; print('✓ All libraries installed')"
```

---

## **Step 2: Implement Adversarial Training Pipeline (60–90 minutes)**

Create:  
`adversarial-training/01_adversarial_training.ipynb`

### **A. Load baseline model from Module 1**

```python
from models.baseline import SimpleConvNet
model = SimpleConvNet().to(device)
```

### **B. Generate adversarial examples (FGSM + PGD)**

Using ART:

```python
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

fgsm = FastGradientMethod(estimator=classifier, eps=0.2)
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.3, max_iter=40)
```

### **C. Retrain model with adversarial samples**

```python
adv_examples = fgsm.generate(x=train_images)
combined = torch.cat([train_images, adv_examples])
```

Train for 3–5 epochs and log:
- Clean accuracy  
- Robust accuracy  

### **D. Save hardened model**

```python
torch.save(model.state_dict(), "../models/hardened_model_v1.pth")
```

---

## **Step 3: Add Differential Privacy (45–60 minutes)**

Create:  
`differential-privacy/02_dp_training.ipynb`

### **A. Wrap optimizer with Opacus**

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
```

### **B. Track privacy budget**

```python
epsilon, best_alpha = privacy_engine.get_privacy_spent(delta=1e-5)
print(f"ε = {epsilon:.2f}, δ = 1e-5")
```

### **C. Compare accuracy vs privacy**

Create a table:

| Model Variant | Clean Acc | Robust Acc | ε (epsilon) |
|---------------|-----------|------------|-------------|
| Baseline | 98% | 12% | N/A |
| Adv‑trained | 96% | 48% | N/A |
| DP‑SGD | 94% | 10% | 3.1 |
| Adv + DP | 92% | 41% | 3.1 |

### **D. Save DP model**

```python
torch.save(model.state_dict(), "../models/dp_model.pth")
```

---

## **Step 4: Create Defense Architecture Diagram (10 minutes)**

File:  
`diagrams/week3_defense_architecture.md`

```
┌──────────────────────────────┐
│     Training Data (DP)       │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  Adversarial Training Loop   │
│   (FGSM / PGD / CW)          │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│     Hardened Model v1        │
└──────────────────────────────┘
```

---

# **WEEK 4 — Robustness Dashboard + Multi‑Cloud Defense Comparison**

---

## **Step 5: Build Robustness Evaluation Dashboard (60 minutes)**

Create:  
`robustness-dashboard/app.py`

### **A. Dashboard sections**

- Attack Success Rate (ASR)  
- Robust accuracy  
- Perturbation sensitivity  
- Privacy budget tracking  
- Cloud‑specific defense comparison  

### **B. Example Streamlit layout**

```python
import streamlit as st
import plotly.express as px
import pandas as pd

st.title("AI Defense Toolkit – Robustness Dashboard")

df = pd.read_csv("metrics/robustness_metrics.csv")
fig = px.bar(df, x="attack", y="success_rate", title="Attack Success Rate")
st.plotly_chart(fig)
```

Run:

```bash
streamlit run app.py
```

---

## **Step 6: Compare Defenses Across Clouds (45 minutes)**

Create:

- `cloud-comparison/aws_bedrock_defense.md`
- `cloud-comparison/azure_openai_defense.md`
- `cloud-comparison/vertex_ai_defense.md`

### Each file includes:

#### **A. Guardrail effectiveness**
- Azure Content Safety  
- Bedrock Guardrails  
- Vertex AI Safety Filters  

#### **B. Logging & monitoring**
- Azure Monitor  
- CloudWatch  
- Cloud Logging  

#### **C. IAM & data protection**
- Entra ID  
- AWS IAM  
- GCP IAM + CMEK  

#### **D. Robustness comparison table**

| Cloud | Guardrails | Robust Acc | Notes |
|-------|------------|------------|-------|
| Azure OpenAI | Strong | 48% | Best logging |
| AWS Bedrock | Medium | 45% | Best isolation |
| Vertex AI | Strong | 47% | Best DP support |

---

## **Step 7: Add Case Studies (30 minutes)**

Folder:  
`docs/case_studies/`

Examples:
- Azure ML adversarial training improvement  
- SageMaker DP‑SGD privacy trade‑offs  
- Vertex AI drift detection  

Each case study includes:
- Problem  
- Attack vector  
- Defense applied  
- Outcome  
- Cloud implications  

---

## **Step 8: Create Model Hardening Playbook (30 minutes)**

File:  
`docs/model_hardening_playbook.md`

Sections:
- Threat model  
- Defensive strategies  
- Adversarial training patterns  
- Differential privacy guidance  
- Cloud‑specific hardening controls  
- Governance mapping (NIST AI RMF, ISO 42001)  

---

# 🎉 **End of Weeks 3–4 Deliverables**

### ✅ **Defense Toolkit Repository**
Includes:
- Adversarial training pipelines  
- Differential privacy examples  
- Robustness evaluation dashboard  
- Cloud defense comparison  
- Architecture diagrams  
- Case studies  

### ✅ **Model Hardening Playbook**
Enterprise‑ready documentation for AI security teams.

---

Just tell me which one you want next.
