# 🛡️ **AI Defense Toolkit – AWS Version (Step-by-Step Walkthrough)**

## 📋 Overview

This walkthrough implements **model hardening** on **AWS**: adversarial training, differential privacy, robustness evaluation, and defense controls using **SageMaker**, **Bedrock**, **S3**, **IAM**, and **CloudWatch**.

**Goals:**
- ✅ Adversarial training (local or SageMaker)
- ✅ Differential privacy (Opacus locally; SageMaker DP-SGD where applicable)
- ✅ Robustness dashboard (Streamlit on EC2 or SageMaker Studio)
- ✅ AWS defense stack: Bedrock Guardrails, IAM, CloudWatch
- ✅ Model Hardening Playbook (AWS-focused)

**Time:** 8–10 hours  
**Prerequisites:** Module 1 baseline model + attacks, Python 3.9+, AWS account, AWS CLI configured  
**Next:** Model Monitoring + AI Guardrails (Module 3)

---

# 🚀 **Step-by-Step Implementation (AWS)**

---

## **Step 1: Project Setup (15 minutes)**

### 1.1 Create project structure (Windows)

```powershell
mkdir module2-defense-toolkit-aws
cd module2-defense-toolkit-aws

mkdir adversarial-training, differential-privacy, robustness-dashboard, cloud-comparison, diagrams, docs\case_studies, models, data
```

### 1.2 AWS CLI and region

```powershell
aws configure
# Set: AWS Access Key, Secret Key, default region (e.g. us-east-1)
aws sts get-caller-identity
```

### 1.3 Create S3 bucket for artifacts (optional but recommended)

```powershell
aws s3 mb s3://YOUR-BUCKET-NAME-ai-defense-toolkit --region us-east-1
# Enable versioning for model/checkpoint rollback
aws s3api put-bucket-versioning --bucket YOUR-BUCKET-NAME-ai-defense-toolkit --versioning-configuration Status=Enabled
```

### 1.4 Python environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 1.5 requirements.txt (add AWS SDKs)

```text
# ML/AI
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

# AWS
boto3>=1.28.0
sagemaker>=2.190.0

# Utilities
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

```powershell
pip install -r requirements.txt
python -c "import torch, art, opacus, boto3; print('All libraries installed')"
```

---

## **Step 2: Adversarial Training Pipeline – AWS Options (60–90 min)**

**Create:** `adversarial-training/01_adversarial_training.ipynb`

### Complete Sample Code for `01_adversarial_training.ipynb`

```python
# Cell 1: Imports and Setup
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import boto3
from botocore.exceptions import ClientError

# Add project root for imports (adjust path as needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import SimpleConvNet

try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
except ImportError:
    print("Install ART: pip install adversarial-robustness-toolbox")
    sys.exit(1)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data"
MODELS_DIR = "../models"
S3_BUCKET = "YOUR-BUCKET-NAME-ai-defense-toolkit"  # Replace with your bucket
BATCH_SIZE = 128
EPOCHS = 3
FGSM_EPS = 0.2
PGD_EPS = 0.3
PGD_STEPS = 10

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# AWS S3 client (optional)
s3_client = boto3.client("s3") if boto3 else None

# Cell 2: Load MNIST Dataset
def get_mnist_loaders():
    """Download MNIST and return train/test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader

train_loader, test_loader = get_mnist_loaders()
print(f"✓ Loaded MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

# Cell 3: Initialize Baseline Model
model = SimpleConvNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Quick baseline train (1 epoch)
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

print("Training baseline (1 epoch)...")
train_epoch(model, train_loader, criterion, optimizer, DEVICE)
clean_acc_baseline = evaluate(model, test_loader, DEVICE)
print(f"Baseline clean accuracy: {clean_acc_baseline:.2%}")

# Cell 4: Wrap Model for ART
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type="gpu" if torch.cuda.is_available() else "cpu",
)
print("✓ Model wrapped for ART")

# Cell 5: Generate Adversarial Examples (FGSM)
train_iter = iter(train_loader)
x_batch, y_batch = next(train_iter)
x_batch = x_batch.to(DEVICE)
x_np = x_batch.detach().cpu().numpy()

fgsm = FastGradientMethod(estimator=classifier, eps=FGSM_EPS)
print(f"Generating FGSM adversarial examples (ε={FGSM_EPS})...")
x_adv_fgsm = fgsm.generate(x=x_np)
x_adv_tensor = torch.from_numpy(x_adv_fgsm).float().to(DEVICE)
print(f"✓ Generated {len(x_adv_tensor)} adversarial examples")

# Optional: Generate PGD examples
pgd = ProjectedGradientDescent(estimator=classifier, eps=PGD_EPS, max_iter=PGD_STEPS)
x_adv_pgd = pgd.generate(x=x_np)
x_adv_pgd_tensor = torch.from_numpy(x_adv_pgd).float().to(DEVICE)
print(f"✓ Generated {len(x_adv_pgd_tensor)} PGD adversarial examples")

# Cell 6: Combine Clean + Adversarial Data
combined_x = torch.cat([x_batch, x_adv_tensor])
combined_y = torch.cat([y_batch, y_batch])
adv_dataset = TensorDataset(combined_x, combined_y)
adv_loader = DataLoader(adv_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"✓ Combined dataset: {len(combined_x)} samples (clean + adversarial)")

# Cell 7: Adversarial Training
print("Adversarial training (FGSM mix)...")
for ep in range(EPOCHS - 1):
    acc = train_epoch(model, adv_loader, criterion, optimizer, DEVICE)
    print(f"  Epoch {ep + 1} train acc: {acc:.2%}")

# Cell 8: Evaluate Robust Accuracy
test_iter = iter(test_loader)
x_test, y_test = next(test_iter)
x_test = x_test.to(DEVICE)
x_test_np = x_test.detach().cpu().numpy()
x_test_adv = fgsm.generate(x=x_test_np)
x_test_adv_t = torch.from_numpy(x_test_adv).float().to(DEVICE)

model.eval()
with torch.no_grad():
    pred_adv = model(x_test_adv_t).argmax(1)
    robust_correct = (pred_adv == y_test.to(DEVICE)).sum().item()
robust_acc = robust_correct / y_test.size(0)
clean_acc_final = evaluate(model, test_loader, DEVICE)

print(f"\n📊 Results:")
print(f"  Clean accuracy: {clean_acc_final:.2%}")
print(f"  Robust accuracy (FGSM ε={FGSM_EPS}): {robust_acc:.2%}")
print(f"  Improvement: {robust_acc - 0.12:.2%} (vs baseline robust ~12%)")

# Cell 9: Save Hardened Model Locally
path = os.path.join(MODELS_DIR, "hardened_model_v1.pth")
torch.save(model.state_dict(), path)
print(f"✓ Saved: {path}")

# Cell 10: Upload to S3 (Optional)
if s3_client:
    try:
        s3_key = "models/hardened_model_v1.pth"
        s3_client.upload_file(path, S3_BUCKET, s3_key)
        print(f"✓ Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
    except ClientError as e:
        print(f"⚠ S3 upload failed: {e}")
else:
    print("⚠ S3 client not configured; skipping upload")
```

### Option B: Run on SageMaker (AWS-native)

**Create:** `adversarial-training/train_adversarial.py` (same code as above, but adapted for SageMaker)

**SageMaker Training Job:**

```python
import sagemaker
from sagemaker.pytorch import PyTorch

role = sagemaker.get_execution_role()  # or pass IAM role ARN
sess = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_adversarial.py",
    source_dir="adversarial-training",
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",  # GPU for faster training
    framework_version="2.0",
    py_version="py310",
    hyperparameters={"epochs": 5, "eps": 0.2},
)

# Train; model artifact goes to S3 automatically
estimator.fit({"training": "s3://YOUR-BUCKET/data/train/"})
```

**AWS notes:**
- Use **SageMaker Training Jobs** for reproducible, logged runs (CloudWatch Logs).
- Store datasets in **S3**; use **SageMaker input channels** or `s3://` in your script.
- IAM role for the training job should have `s3:GetObject` on data bucket and `s3:PutObject` on the model bucket.

---

## **Step 3: Differential Privacy – AWS Version (45–60 min)**

**Create:** `differential-privacy/02_dp_training.ipynb`

### Complete Sample Code for `02_dp_training.ipynb`

```python
# Cell 1: Imports and Setup
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import boto3
import json
import time
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import SimpleConvNet

try:
    from opacus import PrivacyEngine
except ImportError:
    print("Install Opacus: pip install opacus")
    sys.exit(1)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data"
MODELS_DIR = "../models"
S3_BUCKET = "YOUR-BUCKET-NAME-ai-defense-toolkit"  # Replace with your bucket
CLOUDWATCH_LOG_GROUP = "/ai-defense/dp-training"  # Optional CloudWatch log group
BATCH_SIZE = 256
EPOCHS = 2
NOISE_MULTIPLIER = 1.1
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

os.makedirs(MODELS_DIR, exist_ok=True)

# AWS clients (optional)
s3_client = boto3.client("s3") if boto3 else None
logs_client = boto3.client("logs") if boto3 else None

# Cell 2: Load MNIST Dataset
def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader

train_loader, test_loader = get_mnist_loaders()
print(f"✓ Loaded MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

# Cell 3: Initialize Model
model = SimpleConvNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Cell 4: Wrap with Opacus PrivacyEngine (DP-SGD)
privacy_engine = PrivacyEngine()
print(f"Wrapping model with DP-SGD (noise_multiplier={NOISE_MULTIPLIER}, max_grad_norm={MAX_GRAD_NORM})...")

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
)
print("✓ Model wrapped with Opacus PrivacyEngine")

# Cell 5: DP-SGD Training Loop
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

print("DP-SGD training...")
for ep in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total
    epsilon = privacy_engine.get_epsilon(delta=DELTA)
    print(f"  Epoch {ep + 1} train acc: {acc:.2%} | ε ≈ {epsilon:.2f}")

# Cell 6: Get Final Privacy Spent
epsilon, best_alpha = privacy_engine.get_privacy_spent(delta=DELTA)
print(f"\n📊 Privacy Spent:")
print(f"  ε (epsilon) = {epsilon:.2f}")
print(f"  δ (delta) = {DELTA}")
print(f"  Best α = {best_alpha:.2f}")

# Cell 7: Evaluate Test Accuracy
test_acc = evaluate(model, test_loader, DEVICE)
print(f"\n📊 Test Accuracy: {test_acc:.2%}")

# Cell 8: Save DP Model Locally
path = os.path.join(MODELS_DIR, "dp_model.pth")
torch.save(model.state_dict(), path)
print(f"✓ Saved: {path}")

# Cell 9: Log ε to CloudWatch (Optional)
if logs_client:
    try:
        # Create log group if it doesn't exist
        try:
            logs_client.create_log_group(logGroupName=CLOUDWATCH_LOG_GROUP)
        except logs_client.exceptions.ResourceAlreadyExistsException:
            pass
        
        log_stream = f"dp-model-{int(time.time())}"
        logs_client.create_log_stream(
            logGroupName=CLOUDWATCH_LOG_GROUP,
            logStreamName=log_stream
        )
        
        logs_client.put_log_events(
            logGroupName=CLOUDWATCH_LOG_GROUP,
            logStreamName=log_stream,
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": json.dumps({
                    "epsilon": epsilon,
                    "delta": DELTA,
                    "noise_multiplier": NOISE_MULTIPLIER,
                    "test_accuracy": test_acc,
                    "epochs": EPOCHS
                })
            }]
        )
        print(f"✓ Logged to CloudWatch: {CLOUDWATCH_LOG_GROUP}/{log_stream}")
    except ClientError as e:
        print(f"⚠ CloudWatch logging failed: {e}")

# Cell 10: Upload to S3 (Optional)
if s3_client:
    try:
        s3_key = "models/dp_model.pth"
        s3_client.upload_file(path, S3_BUCKET, s3_key)
        print(f"✓ Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
    except ClientError as e:
        print(f"⚠ S3 upload failed: {e}")
else:
    print("⚠ S3 client not configured; skipping upload")
```

### Option B: SageMaker and DP-SGD

- SageMaker doesn’t ship Opacus in the default image; use a **custom training image** or **bring your own container** with Opacus installed and run the same Opacus script as a SageMaker training job.
- Log ε to **CloudWatch** via `boto3` or print to stdout (SageMaker captures it).

### Accuracy vs privacy table (example)

| Model Variant   | Clean Acc | Robust Acc | ε (epsilon) |
|-----------------|-----------|------------|-------------|
| Baseline        | 98%       | 12%        | N/A         |
| Adv-trained     | 96%       | 48%        | N/A         |
| DP-SGD          | 94%       | 10%        | 3.1         |
| Adv + DP        | 92%       | 41%        | 3.1         |

---

## **Step 4: Defense Architecture Diagram – AWS (10 min)**

**File:** `diagrams/week3_defense_architecture_aws.md`

```text
┌─────────────────────────────────────┐
│  Training Data (S3)                 │
│  Optional: SageMaker Data Wrangler  │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Adversarial Training               │
│  (FGSM / PGD) – Local or SageMaker  │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Hardened Model v1                  │
│  Stored in S3 (+ optional registry) │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Deployment: SageMaker Endpoint     │
│  + Bedrock Guardrails (if LLM)      │
└─────────────────────────────────────┘
```

---

## **Step 5: Robustness Evaluation Dashboard – AWS (60 min)**

**Create:** `robustness-dashboard/app.py`

### Complete Sample Code for `app.py`

```python
"""
AI Defense Toolkit – Robustness Dashboard (Streamlit demo with AWS integration).
Run: streamlit run app.py   (from robustness_dashboard/ directory)
"""
import os
import boto3
from botocore.exceptions import ClientError

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
S3_BUCKET = "YOUR-BUCKET-NAME-ai-defense-toolkit"  # Replace with your bucket
S3_METRICS_KEY = "metrics/robustness_metrics.csv"
S3_COMPARISON_KEY = "metrics/model_comparison.csv"

# Initialize AWS clients (optional)
s3_client = None
try:
    s3_client = boto3.client("s3")
except Exception:
    pass


@st.cache_data
def load_robustness_metrics():
    """Load metrics from S3 or local file."""
    # Try S3 first
    if s3_client:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_METRICS_KEY)
            return pd.read_csv(obj["Body"])
        except ClientError:
            pass
    
    # Fallback to local file
    path = os.path.join(METRICS_DIR, "robustness_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Default fallback
    return pd.DataFrame({
        "attack": ["FGSM", "PGD"],
        "success_rate": [0.88, 0.95],
        "model_variant": ["baseline", "baseline"],
    })


@st.cache_data
def load_model_comparison():
    """Load model comparison from S3 or local file."""
    # Try S3 first
    if s3_client:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_COMPARISON_KEY)
            return pd.read_csv(obj["Body"])
        except ClientError:
            pass
    
    # Fallback to local file
    path = os.path.join(METRICS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # Default fallback
    return pd.DataFrame({
        "model_variant": ["Baseline", "Adv-trained", "DP-SGD"],
        "clean_acc": [0.98, 0.96, 0.94],
        "robust_acc": [0.12, 0.48, 0.10],
        "epsilon": ["N/A", "N/A", "3.1"],
    })


def main():
    st.set_page_config(page_title="AI Defense Toolkit", page_icon="🛡️", layout="wide")
    st.title("🛡️ AI Defense Toolkit – Robustness Dashboard (AWS)")
    st.caption("Demo: Attack Success Rate, Robust Accuracy, Model Comparison, Privacy Budget")
    
    # AWS status indicator
    if s3_client:
        st.sidebar.success("✓ AWS S3 Connected")
    else:
        st.sidebar.info("⚠ Using local metrics (S3 not configured)")

    df = load_robustness_metrics()
    df_models = load_model_comparison()

    # --- Attack Success Rate ---
    st.header("1. Attack Success Rate (ASR)")
    if not df.empty and "success_rate" in df.columns:
        fig_asr = px.bar(
            df,
            x="attack",
            y="success_rate",
            color="model_variant",
            barmode="group",
            title="Attack Success Rate by Attack Type and Model",
            labels={"success_rate": "Attack Success Rate", "attack": "Attack Type"},
        )
        st.plotly_chart(fig_asr, use_container_width=True)
    else:
        st.info("Add robustness_metrics.csv with columns: attack, success_rate, model_variant")

    # --- Model comparison: Clean vs Robust ---
    st.header("2. Model Comparison (Clean vs Robust Accuracy)")
    if not df_models.empty:
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(name="Clean Acc", x=df_models["model_variant"], y=df_models["clean_acc"])
        )
        fig_comp.add_trace(
            go.Bar(name="Robust Acc", x=df_models["model_variant"], y=df_models["robust_acc"])
        )
        fig_comp.update_layout(
            barmode="group",
            title="Clean vs Robust Accuracy by Model",
            xaxis_title="Model Variant",
            yaxis_title="Accuracy",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Add model_comparison.csv")

    # --- Privacy budget ---
    st.header("3. Privacy Budget (ε)")
    if not df_models.empty and "epsilon" in df_models.columns:
        eps_df = df_models[df_models["epsilon"].astype(str) != "N/A"].copy()
        if not eps_df.empty:
            eps_df["epsilon"] = pd.to_numeric(eps_df["epsilon"], errors="coerce")
            fig_eps = px.bar(
                eps_df.dropna(subset=["epsilon"]),
                x="model_variant",
                y="epsilon",
                title="Privacy Budget (ε) for DP Models",
                labels={"epsilon": "ε (epsilon)", "model_variant": "Model Variant"},
            )
            st.plotly_chart(fig_eps, use_container_width=True)
        else:
            st.metric("Privacy budget", "N/A (no DP models in table)")
    else:
        st.metric("Privacy budget", "ε = 3.1 (example for DP-SGD)")

    # --- Summary table ---
    st.header("4. Summary Table")
    if not df_models.empty:
        st.dataframe(df_models, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            pd.DataFrame({
                "Model": ["Baseline", "Adv-trained", "DP-SGD", "Adv + DP"],
                "Clean Acc": ["98%", "96%", "94%", "92%"],
                "Robust Acc": ["12%", "48%", "10%", "41%"],
                "ε": ["N/A", "N/A", "3.1", "3.1"],
            }),
            use_container_width=True,
            hide_index=True,
        )
    
    # --- AWS Integration Info ---
    st.sidebar.header("AWS Integration")
    st.sidebar.markdown("""
    **S3 Metrics:**
    - Loads from `s3://{bucket}/metrics/`
    - Falls back to local files if S3 unavailable
    
    **CloudWatch:**
    - Link to CloudWatch Logs for training jobs
    - View privacy budget logs in `/ai-defense/dp-training`
    """)


if __name__ == "__main__":
    main()
```

**Run locally:** `streamlit run app.py`  
**Run on AWS:** Deploy Streamlit on **EC2** or inside **SageMaker Studio** (custom app); optionally write metrics to S3 from your evaluation scripts.

---

## **Step 6: AWS Defense Stack – Documentation (45 min)**

**Create:** `cloud-comparison/aws_bedrock_defense.md` (and keep it as the main AWS reference)

### Complete Sample Content for `aws_bedrock_defense.md`

```markdown
# AWS Bedrock Defense Stack – Complete Guide

## Overview

This document outlines AWS-specific defense controls for AI model hardening, including Bedrock Guardrails, SageMaker security, S3 data protection, and CloudWatch monitoring.

---

## A. Guardrails

### Bedrock Guardrails

**Bedrock Guardrails** provide content filtering, topic denial, word filters, and PII redaction for LLM deployments.

#### Sample Code: Create Bedrock Guardrail

```python
import boto3
import json

bedrock_client = boto3.client("bedrock", region_name="us-east-1")

# Create a guardrail
guardrail_config = {
    "name": "ai-defense-guardrail",
    "description": "Content filters for AI Defense Toolkit",
    "topicPolicyConfig": {
        "topicsConfig": [
            {
                "name": "harmful_content",
                "definition": "Content that promotes violence, hate, or illegal activities",
                "examples": ["violence", "hate speech", "illegal activities"]
            }
        ]
    },
    "contentPolicyConfig": {
        "filtersConfig": [
            {
                "type": "PROFANITY",
                "inputStrength": "HIGH",
                "outputStrength": "HIGH"
            },
            {
                "type": "HATE",
                "inputStrength": "HIGH",
                "outputStrength": "HIGH"
            },
            {
                "type": "MISCONDUCT",
                "inputStrength": "MEDIUM",
                "outputStrength": "MEDIUM"
            }
        ]
    },
    "wordPolicyConfig": {
        "wordsConfig": [
            {
                "text": "sensitive_data",
                "type": "BLOCKED"
            }
        ]
    },
    "sensitiveInformationPolicyConfig": {
        "piiEntitiesConfig": [
            {"type": "EMAIL"},
            {"type": "PHONE"},
            {"type": "SSN"}
        ],
        "regexesConfig": [
            {
                "name": "credit_card",
                "pattern": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
                "action": "BLOCK"
            }
        ]
    }
}

# Note: Actual API may vary; check AWS Bedrock documentation for current API
# response = bedrock_client.create_guardrail(**guardrail_config)
```

#### Guardrail Effectiveness Mapping

| Threat | Guardrail Rule | Effectiveness |
|--------|---------------|---------------|
| Harmful content | Content filters (HATE, PROFANITY) | High |
| PII leakage | PII entities config | High |
| Sensitive data | Word policy + regexes | Medium |
| Topic violations | Topic policy | High |

---

## B. Logging and Monitoring

### CloudWatch Logs

**SageMaker Training Logs:**

```python
import boto3

logs_client = boto3.client("logs")

# View SageMaker training job logs
log_group = "/aws/sagemaker/TrainingJobs"
log_stream = "your-training-job-name"

response = logs_client.get_log_events(
    logGroupName=log_group,
    logStreamName=log_stream,
    limit=100
)

for event in response["events"]:
    print(f"{event['timestamp']}: {event['message']}")
```

**CloudWatch Metrics:**

```python
cloudwatch = boto3.client("cloudwatch")

# Get SageMaker endpoint metrics
metrics = cloudwatch.get_metric_statistics(
    Namespace="AWS/SageMaker",
    MetricName="ModelLatency",
    Dimensions=[
        {"Name": "EndpointName", "Value": "your-endpoint-name"},
        {"Name": "VariantName", "Value": "AllTraffic"}
    ],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow(),
    Period=300,
    Statistics=["Average", "Maximum"]
)
```

**Bedrock Invocation Logging:**

```python
# Enable Bedrock logging (via AWS Console or CLI)
# aws bedrock put-model-invocation-logging-configuration \
#   --logging-config '{
#     "textDataDeliveryEnabled": true,
#     "imageDataDeliveryEnabled": true,
#     "embeddingDataDeliveryEnabled": true,
#     "cloudWatchConfig": {
#       "logGroupName": "/aws/bedrock/modelinvocations",
#       "roleArn": "arn:aws:iam::ACCOUNT:role/BedrockLoggingRole"
#     }
#   }'
```

---

## C. IAM and Data Protection

### IAM Roles

**SageMaker Training Role (sample policy):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-data-bucket/*",
        "arn:aws:s3:::your-model-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

**Bedrock Invocation Role:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    }
  ]
}
```

### S3 Encryption and Versioning

**Enable S3 encryption and versioning:**

```python
import boto3

s3_client = boto3.client("s3")
bucket_name = "your-bucket-name"

# Enable versioning
s3_client.put_bucket_versioning(
    Bucket=bucket_name,
    VersioningConfiguration={"Status": "Enabled"}
)

# Enable encryption (SSE-S3)
s3_client.put_bucket_encryption(
    Bucket=bucket_name,
    ServerSideEncryptionConfiguration={
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }
)

# Or use KMS (CMEK)
s3_client.put_bucket_encryption(
    Bucket=bucket_name,
    ServerSideEncryptionConfiguration={
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "aws:kms",
                    "KMSMasterKeyID": "arn:aws:kms:region:account:key/key-id"
                }
            }
        ]
    }
)
```

**S3 Bucket Policy (least privilege):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowSageMakerAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}
```

---

## D. Robustness Comparison (AWS-Focused)

| Component           | Control              | Implementation | Notes                    |
|--------------------|----------------------|----------------|--------------------------|
| SageMaker Training | Isolated VPC, IAM    | VPC config in Estimator | Reproducible runs        |
| SageMaker Endpoint | VPC, encryption      | Endpoint config | Hardened model hosting   |
| Bedrock            | Guardrails, logging  | Guardrail API  | LLM safety layer         |
| S3                 | Encryption, versioning| Bucket policies| Model/data protection    |
| CloudWatch         | Logs, metrics        | Auto-enabled   | Monitoring & audit       |

---

## E. Sample Integration: End-to-End Defense

```python
import boto3
import json

# 1. Train model with adversarial training (SageMaker)
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train_adversarial.py",
    role="arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole",
    instance_type="ml.g4dn.xlarge",
    vpc_config={
        "Subnets": ["subnet-xxx"],
        "SecurityGroupIds": ["sg-xxx"]
    }
)
estimator.fit({"training": "s3://your-bucket/data/"})

# 2. Deploy endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="hardened-model-endpoint"
)

# 3. Enable CloudWatch monitoring
cloudwatch = boto3.client("cloudwatch")
cloudwatch.put_metric_alarm(
    AlarmName="HighModelLatency",
    MetricName="ModelLatency",
    Namespace="AWS/SageMaker",
    Statistic="Average",
    Period=300,
    EvaluationPeriods=1,
    Threshold=1000.0,
    ComparisonOperator="GreaterThanThreshold"
)

# 4. Use Bedrock Guardrails (for LLM use cases)
# Associate guardrail with Bedrock model invocation
```

---

## Summary

- **Guardrails:** Content filters, PII redaction, topic denial
- **Logging:** CloudWatch Logs for SageMaker, Bedrock invocation logs
- **IAM:** Least-privilege roles for SageMaker, Bedrock, S3
- **Encryption:** S3 SSE-S3 or SSE-KMS, SageMaker endpoint encryption
- **Monitoring:** CloudWatch Metrics and Alarms for model performance
```

**Save this as:** `cloud-comparison/aws_bedrock_defense.md`

---

## **Step 7: Case Studies – AWS (30 min)**

**Folder:** `docs/case_studies/`

Create at least one AWS-focused case study, e.g. **`docs/case_studies/sagemaker_dp_sgd_tradeoffs.md`**:

- **Problem:** Need DP for a sensitive dataset used in SageMaker.
- **Attack vector:** Membership inference / data extraction.
- **Defense:** DP-SGD (Opacus) in a custom SageMaker container; track ε.
- **Outcome:** Accuracy vs ε table; decision on acceptable ε (e.g. ε &lt; 3).
- **Cloud implications:** Custom image in ECR, IAM for ECR/S3/SageMaker, CloudWatch for ε logs.

You can add more: e.g. “Adversarial training on SageMaker” or “Bedrock Guardrails vs custom filters.”

---

## **Step 8: Model Hardening Playbook – AWS (30 min)**

**File:** `docs/model_hardening_playbook_aws.md`

**Sections:**

1. **Threat model**  
   Same as main playbook; add AWS-specific threats (e.g. S3 exposure, role escalation).

2. **Defensive strategies**  
   Adversarial training, DP, input validation; where each runs (local vs SageMaker).

3. **Adversarial training on AWS**  
   - Data in S3 → SageMaker training job or local script.  
   - Logging to CloudWatch; model artifact to S3.

4. **Differential privacy on AWS**  
   - Opacus in custom SageMaker container; log ε to CloudWatch; store DP model in S3.

5. **AWS hardening controls**  
   - **IAM:** Roles for SageMaker, Bedrock, S3; no long-lived keys in code.  
   - **S3:** Encryption, versioning, least-privilege bucket policies.  
   - **SageMaker:** VPC, endpoint protection, optional VPC endpoints for S3.  
   - **Bedrock:** Guardrails, invocation logging, no PII in prompts where possible.

6. **Governance mapping**  
   Map controls to NIST AI RMF / ISO 42001 and note where AWS (e.g. SOC, ISO) helps.

---

# 🎉 **End of AWS Walkthrough – Deliverables**

### Checklist

- [ ] Project under `module2-defense-toolkit-aws/` with venv and `requirements.txt` (including `boto3`, `sagemaker`)
- [ ] S3 bucket created (and optional versioning); AWS CLI configured
- [ ] `adversarial-training/01_adversarial_training.ipynb` (local and/or SageMaker)
- [ ] `differential-privacy/02_dp_training.ipynb`; DP model saved and optionally uploaded to S3
- [ ] `diagrams/week3_defense_architecture_aws.md`
- [ ] `robustness-dashboard/app.py` (with optional S3 metrics); run locally or on EC2/Studio
- [ ] `cloud-comparison/aws_bedrock_defense.md` (guardrails, logging, IAM, robustness table)
- [ ] At least one AWS case study in `docs/case_studies/`
- [ ] `docs/model_hardening_playbook_aws.md`

### AWS-specific outcomes

- Models and metrics can live in **S3** with encryption and versioning.
- Training can be **reproducible** and **logged** via **SageMaker** and **CloudWatch**.
- **Bedrock Guardrails** and **IAM** provide a clear defense layer for LLM and access control.
- One playbook and one diagram focused on **AWS** for your team or compliance.

---

# 🎬 **Demo Walkthrough with Sample Code**

This section provides **runnable sample code** for a complete demo. All code is available in the `demo/` folder.

**Time:** ~1–2 hours for full demo  
**Prerequisites:** Python 3.9+, AWS CLI configured (optional for local-only demo)

---

## **Demo Step 1: Project Setup (≈10 min)**

### 1.1 Go to demo folder

```powershell
cd c:\Users\Elisha\playground\AI_Model_Defense_Hardening_Toolkit\demo
```

### 1.2 Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 1.3 Install dependencies (includes AWS SDKs)

```powershell
pip install -r requirements.txt
```

**`demo/requirements.txt`** includes:

```text
torch>=2.0.0
torchvision>=0.15.0
art>=1.14.0
opacus>=1.4.0
streamlit>=1.29.0
plotly>=5.17.0
boto3>=1.28.0
sagemaker>=2.190.0
```

### 1.4 Verify installation

```powershell
python -c "import torch, art, opacus, boto3; print('All libraries OK')"
```

### 1.5 (Optional) Configure AWS and create S3 bucket

```powershell
aws configure
aws s3 mb s3://YOUR-BUCKET-NAME-ai-defense-demo --region us-east-1
aws s3api put-bucket-versioning --bucket YOUR-BUCKET-NAME-ai-defense-demo --versioning-configuration Status=Enabled
```

---

## **Demo Step 2: Adversarial Training with Sample Code (≈15–20 min)**

### 2.1 Run adversarial training script

From **repo root**:

```powershell
cd c:\Users\Elisha\playground\AI_Model_Defense_Hardening_Toolkit
python -m demo.adversarial_training.train_adversarial
```

**What it does:**
1. Downloads MNIST into `demo/data/`
2. Trains baseline CNN (1 epoch)
3. Wraps model with ART and generates FGSM adversarial examples
4. Trains on clean + adversarial data for 2 more epochs
5. Reports clean and robust accuracy
6. Saves `demo/models/hardened_model_v1.pth`

### 2.2 Sample code (core logic)

**`demo/adversarial_training/train_adversarial.py`** includes:

```python
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import torch

# Wrap model for ART
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type="gpu" if torch.cuda.is_available() else "cpu",
)

# Generate adversarial examples
fgsm = FastGradientMethod(estimator=classifier, eps=0.2)
x_adv = fgsm.generate(x=x_np)

# Combine clean + adversarial
combined_x = torch.cat([x_clean, torch.from_numpy(x_adv).float()])
combined_y = torch.cat([y_batch, y_batch])

# Train on combined dataset
# ... training loop ...
torch.save(model.state_dict(), "models/hardened_model_v1.pth")
```

### 2.3 (Optional) Upload to S3

```python
import boto3
s3 = boto3.client("s3")
s3.upload_file("demo/models/hardened_model_v1.pth", "YOUR-BUCKET", "models/hardened_model_v1.pth")
```

---

## **Demo Step 3: Differential Privacy with Sample Code (≈10–15 min)**

### 3.1 Run DP training

From **repo root**:

```powershell
cd c:\Users\Elisha\playground\AI_Model_Defense_Hardening_Toolkit
python -m demo.differential_privacy.train_dp
```

**What it does:**
1. Loads MNIST
2. Wraps model/optimizer with Opacus `PrivacyEngine`
3. Trains with DP-SGD for 2 epochs
4. Prints spent ε (epsilon) and test accuracy
5. Saves `demo/models/dp_model.pth`

### 3.2 Sample code (core logic)

**`demo/differential_privacy/train_dp.py`** includes:

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

# Train as usual
for epoch in range(EPOCHS):
    for x, y in train_loader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

epsilon, _ = privacy_engine.get_privacy_spent(delta=1e-5)
print(f"Privacy spent: ε = {epsilon:.2f}")

torch.save(model.state_dict(), "models/dp_model.pth")
```

### 3.3 (Optional) Log ε to CloudWatch and upload to S3

```python
import boto3
import json

# Log to CloudWatch
logs = boto3.client("logs")
logs.put_log_events(
    logGroupName="/ai-defense/dp-training",
    logStreamName="dp-model-v1",
    logEvents=[{
        "timestamp": int(time.time() * 1000),
        "message": json.dumps({"epsilon": epsilon, "delta": 1e-5})
    }]
)

# Upload to S3
s3.upload_file("demo/models/dp_model.pth", "YOUR-BUCKET", "models/dp_model.pth")
```

---

## **Demo Step 4: Robustness Dashboard (≈5 min)**

### 4.1 Start Streamlit app

From **demo** folder:

```powershell
cd c:\Users\Elisha\playground\AI_Model_Defense_Hardening_Toolkit\demo\robustness_dashboard
streamlit run app.py
```

Browser opens at `http://localhost:8501`. Dashboard shows:
- **Attack Success Rate (ASR)** by attack and model variant
- **Clean vs Robust Accuracy** bar chart
- **Privacy budget (ε)** for DP models
- **Summary table** (from `metrics/model_comparison.csv`)

### 4.2 Sample metrics (pre-filled)

**`demo/robustness_dashboard/metrics/robustness_metrics.csv`:**

```csv
attack,success_rate,robust_acc,epsilon,model_variant
FGSM_0.1,0.72,0.28,N/A,baseline
FGSM_0.2,0.88,0.12,N/A,baseline
PGD_0.3,0.95,0.05,N/A,baseline
FGSM_0.1,0.38,0.62,N/A,adv_trained
FGSM_0.2,0.52,0.48,N/A,adv_trained
PGD_0.3,0.61,0.39,N/A,adv_trained
FGSM_0.1,0.90,0.10,3.1,dp_sgd
FGSM_0.2,0.92,0.08,3.1,dp_sgd
PGD_0.3,0.94,0.06,3.1,dp_sgd
FGSM_0.1,0.59,0.41,3.1,adv_plus_dp
FGSM_0.2,0.65,0.35,3.1,adv_plus_dp
PGD_0.3,0.72,0.28,3.1,adv_plus_dp
```

**`demo/robustness_dashboard/metrics/model_comparison.csv`:**

```csv
model_variant,clean_acc,robust_acc,epsilon
Baseline,0.98,0.12,N/A
Adv-trained,0.96,0.48,N/A
DP-SGD,0.94,0.10,3.1
Adv + DP,0.92,0.41,3.1
```

### 4.3 (Optional) Load metrics from S3

**`demo/robustness_dashboard/app.py`** can be extended to read from S3:

```python
import boto3
import pandas as pd

@st.cache_data
def load_metrics_from_s3():
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket="YOUR-BUCKET", Key="metrics/robustness_metrics.csv")
        return pd.read_csv(obj["Body"])
    except Exception:
        return pd.read_csv("metrics/robustness_metrics.csv")
```

---

## **Demo Step 5: Defense Architecture Diagram**

**File:** `demo/diagrams/week3_defense_architecture.md`

```text
┌──────────────────────────────┐
│     Training Data (DP)        │
│     (e.g. MNIST on disk)     │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  Adversarial Training Loop   │
│   (FGSM / PGD)               │
│   Clean + Adv batches        │
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│     Hardened Model v1        │
│     (models/hardened_*.pth)  │
└──────────────────────────────┘
```

**AWS version:** Add S3 storage and SageMaker endpoints to the diagram.

---

## **Demo Step 6: Case Study and Playbook**

- **Case study:** `demo/docs/case_studies/example_case_study.md` — problem, attack (FGSM), defense (adversarial training), outcome, AWS implications.
- **Playbook:** `demo/docs/model_hardening_playbook.md` — threat model, strategies, adversarial training patterns, DP guidance, AWS controls, governance mapping.

---

## **Quick Demo Run Order**

1. **Setup:** `cd demo` → `python -m venv venv` → `.\venv\Scripts\activate` → `pip install -r requirements.txt`
2. **Adversarial:** From repo root: `python -m demo.adversarial_training.train_adversarial`
3. **DP:** From repo root: `python -m demo.differential_privacy.train_dp`
4. **Dashboard:** `cd demo\robustness_dashboard` → `streamlit run app.py`
5. **(Optional) AWS:** Upload models to S3, log metrics to CloudWatch
6. **Show:** Architecture diagram, case study, playbook

---

## **Demo File Layout**

```text
demo/
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── baseline.py          # SimpleConvNet for MNIST
│   ├── hardened_model_v1.pth # after adversarial training
│   └── dp_model.pth          # after DP training
├── data/                     # MNIST downloaded here
├── adversarial_training/
│   ├── __init__.py
│   └── train_adversarial.py
├── differential_privacy/
│   ├── __init__.py
│   └── train_dp.py
├── robustness_dashboard/
│   ├── app.py
│   └── metrics/
│       ├── robustness_metrics.csv
│       └── model_comparison.csv
├── diagrams/
│   └── week3_defense_architecture.md
└── docs/
    ├── case_studies/
    │   └── example_case_study.md
    └── model_hardening_playbook.md
```

---

## **Troubleshooting**

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: models` | Run scripts from **repo root** with `python -m demo.adversarial_training.train_adversarial` |
| ART / Opacus import error | `pip install adversarial-robustness-toolbox opacus` |
| Streamlit not found | `pip install streamlit` then `streamlit run app.py` |
| AWS credentials error | Run `aws configure` or set environment variables |
| S3 upload fails | Check IAM permissions: `s3:PutObject` on bucket |

---

**Next:** Apply the same robustness and hardening ideas in Module 3 (Model Monitoring + AI Guardrails), using CloudWatch and Bedrock Guardrails as the primary AWS building blocks.
