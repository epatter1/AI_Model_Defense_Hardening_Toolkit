# **Weeks 3–4 — Module 2: AI Defense Toolkit (Adversarial Training, DP, Robustness)**

This is Module 2 of 5 of the [12-week Security Architect Program](https://github.com/epatter1/AI_Data_Security_Architect_Program/blob/main/12_week_overview.md)

### Focus areas
* #### Implement adversarial training pipelines
* #### Add differential privacy examples
* #### Build robustness evaluation dashboard
* #### Compare defenses across clouds

#### Deliverables:
* Defense Toolkit Repo + Model Hardening Playbook: [Link to walkthrough](https://github.com/epatter1/AI_Model_Defense_Hardening_Toolkit/blob/main/AI_Model_Defense_Hardening_walkthough.md)

---

## **Technologies, Frameworks & Cloud Services Used**

### 🧠 **AI/ML, Adversarial ML & Differential Privacy**

<a href="PyTorch">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
</a>
<a href="TensorFlow">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" />
</a>
<a href="Opacus">
  <img src="https://img.shields.io/badge/Opacus-000000?logo=pytorch&logoColor=white" />
</a>
<a href="TensorFlow-Privacy">
  <img src="https://img.shields.io/badge/TF_Privacy-FF6F00?logo=tensorflow&logoColor=white" />
</a>
<a href="ART">
  <img src="https://img.shields.io/badge/Adversarial_Robustness_Toolbox-005C5C" />
</a>
<a href="CleverHans">
  <img src="https://img.shields.io/badge/CleverHans-4B0082" />
</a>
<a href="Foolbox">
  <img src="https://img.shields.io/badge/Foolbox-DC143C" />
</a>

---

### 🔐 **Security, Threat Modeling & Risk**

<a href="MITRE-ATLAS">
  <img src="https://img.shields.io/badge/MITRE-ATLAS-blue" />
</a>
<a href="OWASP-LLM-Top-10">
  <img src="https://img.shields.io/badge/OWASP-Top_10_for_LLMs-000000?logo=owasp&logoColor=white" />
</a>
<a href="STRIDE">
  <img src="https://img.shields.io/badge/Threat_Modeling-STRIDE-purple" />
</a>
<a href="DP">
  <img src="https://img.shields.io/badge/Differential_Privacy-6A5ACD" />
</a>
<a href="Robustness">
  <img src="https://img.shields.io/badge/Robustness-Evaluation-green" />
</a>

---

## ☁️ **Cloud Platforms**

### **AWS**

<a href="AWS">
  <img src="https://img.shields.io/badge/AWS-232F3E?logo=amazonaws&logoColor=white" />
</a>
<a href="Bedrock">
  <img src="https://img.shields.io/badge/AWS-Bedrock-FF9900?logo=amazonaws&logoColor=white" />
</a>
<a href="SageMaker">
  <img src="https://img.shields.io/badge/Amazon-SageMaker-1F72B8?logo=amazonaws&logoColor=white" />
</a>

---

### **Azure**

<a href="Azure">
  <img src="https://img.shields.io/badge/Azure-0078D4?logo=microsoftazure&logoColor=white" />
</a>
<a href="Azure-ML">
  <img src="https://img.shields.io/badge/Azure_ML-0078D4?logo=microsoftazure&logoColor=white" />
</a>
<a href="Azure-OpenAI">
  <img src="https://img.shields.io/badge/Azure_OpenAI-0078D4?logo=microsoftazure&logoColor=white" />
</a>

---

### **GCP**

<a href="GCP">
  <img src="https://img.shields.io/badge/GCP-4285F4?logo=googlecloud&logoColor=white" />
</a>
<a href="Vertex-AI">
  <img src="https://img.shields.io/badge/Vertex_AI-4285F4?logo=googlecloud&logoColor=white" />
</a>

---

## 🛠️ **DevOps, Infrastructure & Tooling**

<a href="Python">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
</a>
<a href="Docker">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
</a>
<a href="Terraform">
  <img src="https://img.shields.io/badge/Terraform-844FBA?logo=terraform&logoColor=white" />
</a>
<a href="Streamlit">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
</a>

---

## 📊 **Documentation & Visualization**

<a href="Markdown">
  <img src="https://img.shields.io/badge/Markdown-000000?logo=markdown&logoColor=white" />
</a>
<a href="Architecture-Diagrams">
  <img src="https://img.shields.io/badge/Architecture_Diagrams-4285F4" />
</a>
<a href="Playbooks">
  <img src="https://img.shields.io/badge/Model_Hardening-Playbook-green" />
</a>
<a href="Dashboards">
  <img src="https://img.shields.io/badge/Robustness-Dashboard-6A5ACD" />
</a>

---

# **WEEK 3 — Adversarial Training + Differential Privacy Foundations**

### **1. Set Up the Defense Toolkit Structure**
```
module2-defense-toolkit/
    adversarial-training/
    differential-privacy/
    robustness-dashboard/
    cloud-comparison/
    diagrams/
    docs/
```

---

### **2. Implement Adversarial Training Pipelines**
Notebook: `adversarial-training/01_adversarial_training.ipynb`

Includes:
- FGSM, PGD, CW attack generation  
- Retraining models with adversarial examples  
- Measuring robust accuracy  
- Cloud variations (Azure ML, SageMaker, Vertex AI Workbench)

---

### **3. Add Differential Privacy to Training**
Notebook: `differential-privacy/02_dp_training.ipynb`

You will:
- Implement DP‑SGD (Opacus or TF Privacy)  
- Track privacy budget (ε, δ)  
- Compare accuracy vs privacy trade‑offs  
- Document implications for regulated data  

---

### **4. Create Initial Defense Architecture Diagram**
Diagram: `diagrams/week3_defense_architecture.md`

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

### **1. Build the Robustness Evaluation Dashboard**
Folder: `robustness-dashboard/`

Dashboard features:
- Attack success rate (ASR)  
- Robust accuracy  
- Perturbation sensitivity  
- Privacy budget consumption  
- Cloud‑specific performance metrics  

---

### **2. Compare Defenses Across Clouds**
Files:
- `cloud-comparison/aws_bedrock_defense.md`  
- `cloud-comparison/azure_openai_defense.md`  
- `cloud-comparison/vertex_ai_defense.md`  

Each includes:
- Guardrail effectiveness  
- Logging & monitoring maturity  
- Network isolation  
- IAM & data protection  
- Robustness before/after adversarial training  

---

### **3. Add Case Studies + Documentation**
Folder: `docs/case_studies/`

Examples:
- Azure ML adversarial training improvement  
- SageMaker DP‑SGD privacy trade‑offs  
- Vertex AI robustness drift detection  

---

### **4. Create the Model Hardening Playbook**
File: `docs/model_hardening_playbook.md`

Sections:
- Threat model  
- Defensive strategies  
- Adversarial training patterns  
- Differential privacy guidance  
- Cloud‑specific hardening controls  
- Governance mapping (NIST AI RMF, ISO 42001)  

---

## **End of Weeks 3–4 Deliverables**

### ✅ **Defense Toolkit Repository**
Includes:
- Adversarial training pipelines  
- Differential privacy examples  
- Robustness evaluation dashboard  
- Cloud defense comparison  
- Architecture diagrams  
- Case studies  

### ✅ **Model Hardening Playbook**
A polished, enterprise‑ready guide for securing and evaluating AI models.
