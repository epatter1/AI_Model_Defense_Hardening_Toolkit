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

### Option A: Run locally (same as main walkthrough)

- Load baseline model, wrap with ART `PyTorchClassifier`, generate FGSM/PGD examples, retrain on clean + adversarial data, save `hardened_model_v1.pth`.

### Option B: Run on SageMaker (AWS-native)

1. **Package training script** (e.g. `adversarial-training/train_adversarial.py`) that:
   - Loads data from S3 or uses SageMaker input channels
   - Runs adversarial training (FGSM/PGD + retrain)
   - Saves model to `model_dir` (SageMaker copies it to S3)

2. **Use SageMaker PyTorch Estimator:**

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

# Train; model artifact goes to S3
estimator.fit({"training": "s3://YOUR-BUCKET/data/train/"})
```

3. **Upload hardened model to S3:**

```python
# After local training:
import boto3
s3 = boto3.client("s3")
s3.upload_file("../models/hardened_model_v1.pth", "YOUR-BUCKET", "models/hardened_model_v1.pth")
```

**AWS notes:**
- Use **SageMaker Training Jobs** for reproducible, logged runs (CloudWatch Logs).
- Store datasets in **S3**; use **SageMaker input channels** or `s3://` in your script.
- IAM role for the training job should have `s3:GetObject` on data bucket and `s3:PutObject` on the model bucket.

---

## **Step 3: Differential Privacy – AWS Version (45–60 min)**

**Create:** `differential-privacy/02_dp_training.ipynb`

### Option A: Local with Opacus (same as main walkthrough)

- Use `PrivacyEngine.make_private()`, train, track ε with `get_privacy_spent(delta=1e-5)`, save `dp_model.pth`.

### Option B: SageMaker and DP-SGD

- SageMaker doesn’t ship Opacus in the default image; use a **custom training image** or **bring your own container** with Opacus installed and run the same Opacus script as a SageMaker training job.
- Log ε to **CloudWatch** via `boto3` or print to stdout (SageMaker captures it).

### Upload DP model to S3

```python
s3.upload_file("../models/dp_model.pth", "YOUR-BUCKET", "models/dp_model.pth")
```

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

1. **Dashboard sections (same as main):**
   - Attack Success Rate (ASR)
   - Robust accuracy
   - Perturbation sensitivity
   - Privacy budget (ε)
   - **AWS:** link to CloudWatch metrics or SageMaker training job IDs

2. **Optional: Load metrics from S3**

```python
import streamlit as st
import pandas as pd
import boto3

# Optional: read metrics from S3
@st.cache_data
def load_metrics():
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket="YOUR-BUCKET", Key="metrics/robustness_metrics.csv")
        return pd.read_csv(obj["Body"])
    except Exception:
        return pd.read_csv("metrics/robustness_metrics.csv")

df = load_metrics()
st.title("AI Defense Toolkit – Robustness Dashboard (AWS)")
fig = px.bar(df, x="attack", y="success_rate", title="Attack Success Rate")
st.plotly_chart(fig)
```

3. **Run locally:** `streamlit run app.py`  
4. **Run on AWS:** Deploy Streamlit on **EC2** or inside **SageMaker Studio** (custom app); optionally write metrics to S3 from your evaluation scripts.

---

## **Step 6: AWS Defense Stack – Documentation (45 min)**

**Create:** `cloud-comparison/aws_bedrock_defense.md` (and keep it as the main AWS reference)

### A. Guardrails

- **Bedrock Guardrails:** Content filters, topic denial, word filters, PII redaction.
- **Guardrail effectiveness:** Document which guardrail rules you use (e.g. block list, content filters) and how they map to your threat model.

### B. Logging and monitoring

- **CloudWatch Logs:** SageMaker training logs, endpoint invocation logs.
- **CloudWatch Metrics:** Invocation count, latency, errors per endpoint.
- **Bedrock:** Enable model invocation logging (to S3/CloudWatch as per AWS docs) for audit.

### C. IAM and data protection

- **IAM:** Least-privilege roles for SageMaker training/inference, S3, Bedrock.
- **S3:** Bucket policies, encryption (SSE-S3 or SSE-KMS), versioning for model/data.
- **KMS:** Use CMEK for S3 and (where supported) SageMaker/Bedrock for data-at-rest.

### D. Robustness comparison (AWS-focused)

| Component           | Control              | Notes                    |
|--------------------|----------------------|--------------------------|
| SageMaker Training | Isolated VPC, IAM    | Reproducible runs        |
| SageMaker Endpoint | VPC, encryption      | Hardened model hosting   |
| Bedrock            | Guardrails, logging  | LLM safety layer         |
| S3                 | Encryption, versioning| Model/data protection    |

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

**Next:** Apply the same robustness and hardening ideas in Module 3 (Model Monitoring + AI Guardrails), using CloudWatch and Bedrock Guardrails as the primary AWS building blocks.
