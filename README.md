# Geulgyeol — Automated Korean Essay Scoring Agent

![License](https://img.shields.io/github/license/ghko99/essay-agent)
![Stars](https://img.shields.io/github/stars/ghko99/essay-agent?style=social)
![Issues](https://img.shields.io/github/issues/ghko99/essay-agent)

> **Geulgyeol** (Korean for "Texture of Text") is an Automated Essay Scoring Agent. It evaluates essays using a LoRA fine-tuned model and a tool-augmented verification agent.

**Demo**: https://geulgyeol.tech

<img width="1182" height="941" alt="Screenshot" src="https://github.com/user-attachments/assets/7204bae3-17f9-4733-a417-0dc6a5607cf3" />

## 🌟 Introduction

**Geulgyeol** is an automated scoring service that evaluates student essays based on 8 rubrics and provides detailed feedback with evidence. Rather than just assigning a score, an AI agent utilizes real-world tools—such as a spell checker, dictionary, and language analysis API—to measure the text and present a comprehensive report explaining **why** the score was given.

## ⚙️ Scoring Pipeline

```text
Essay Input
   │
   ▼
① Ensemble Scoring (LoRA Fine-tuned Model)
   └─ 8 Rubrics × 1~9 Points, Weighted Self-Consistency Ensemble
   │
   ▼
② Agent Verification (Base Model + External Tools)
   └─ Independently measures keyword fulfillment, spelling, vocabulary validity, 
      and technical terms via tools → Evidence-based independent scoring → Score calibration (±3)
   │
   ▼
③ Final Report
   └─ Overall Review · Strengths · Areas for Improvement · Rubric Scores and Calibration Evidence
```
<img width="1240" height="943" alt="Screenshot" src="https://github.com/user-attachments/assets/7783ca8f-6a47-47f4-8496-5d397ad635aa" />


## 📊 Scoring Rubrics (8 Categories)

| Category | Rubric | Evaluation Criteria |
|---|---|---|
| Task | Task Fulfillment | Does the essay meet the prompt's requirements? |
| Content | Clarity of Explanation | Is the explanation clear? |
| Content | Specificity of Explanation | Are there specific evidence and examples? |
| Content | Relevance of Explanation | Is the content relevant to the topic? |
| Organization | Sentence Connectivity | Do sentences and paragraphs flow naturally? |
| Organization | Text Unity | Is the entire text unified under a single topic? |
| Expression | Vocabulary Appropriateness | Is the vocabulary selection appropriate? |
| Expression | Grammatical Appropriateness | Are spelling and grammar correct? |

Each rubric is scored from 1 to 9. Detailed criteria for each level can be found on the **Scoring Criteria** page of the demo site.

<img width="1240" height="940" alt="Screenshot" src="https://github.com/user-attachments/assets/ba8ae3f0-f8b5-4e8c-8a33-96363b323d8e" />


## 🧠 Agent Design Points

Using an 8B model for "score verification" often results in an **anchoring problem** where the model simply copies the original score. Geulgyeol solves this through the following approaches:

- **Complete Score Concealment**: During the verification stage, the LoRA score is completely hidden from the agent. The agent scores independently using only tool measurements.
- **Step-First Scoring**: The agent must write the evidence first → determine the level (1-5) → and finalize the score within the range corresponding to that level. This prevents contradictions between evidence and score, as well as middle-value escaping.
- **Post-Calibration Limits**: Independent scoring results are clamped to within ±3 points of the original LoRA score to ensure stability.
- **Comparison Revealed Only in Report**: The original score and the calibrated score are only compared after the final score is determined, to explain the calibration evidence to the user.

## 🛠 External APIs (Tools)

| Tool | Purpose |
|---|---|
| Keyword Fulfillment | Measures if the prompt's key terms are present in the text |
| Spell Checker (Bareun) | Detects spelling and spacing errors |
| Korean Dictionary | Verifies if the vocabulary consists of real words |
| Tech Term Verification | Checks the accurate use of professional/technical terms |
| ETRI Language Analysis | Morpheme, dependency parsing, and semantic role labeling |
| Perplexity Measurement | Measures sentence naturalness (custom model) |

## 💻 Tech Stack

- **Scoring Model**: [kanana-1.5-8b-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505) + LoRA Adapter (Fine-tuned on AI Hub Essay/Descriptive Writing Data)
- **Inference Server**: vLLM (AsyncLLMEngine, Request-level LoRA on/off — LoRA ON for scoring, Base Model for the agent)
- **Backend**: FastAPI (Fully async, SSE Streaming)
- **Frontend**: Vanilla JS SPA (No framework)
- **Deployment**: Local GPU Server (RTX 4090) + Cloudflare Tunnel

## 🚀 Getting Started

```bash
# 1. Install Dependencies (Requires Python 3.10 and CUDA GPU)
pip install -r requirements.txt

# 2. Prepare Base Model (Download from HuggingFace)
#    Default path: /home/<user>/models/kanana (Can be changed via KANANA_BASE env variable)

# 3. Configure External API Keys (Create a .env file)
#    OPENDICT_API_KEY, KRD_API_KEY, KTERM_API_KEY,
#    ETRI_API_KEY, ETRI_WISENLU_URL, BAREUN_API_KEY

# 4. Run Server
KANANA_BASE=/path/to/kanana ./run.sh
# → http://localhost:8000
```

The LoRA adapter (`adapter/`) is managed with Git LFS. You must run `git lfs pull` after cloning.

## 📁 Project Structure

```text
essay_agent/
├── backend/
│   ├── main.py        # FastAPI endpoints (Scoring, Verification, Report SSE)
│   ├── model.py       # vLLM Engine, Ensemble Scoring, LoRA Toggle
│   ├── agent.py       # Verification Agent (Tool Loop, Independent Scoring, Report)
│   ├── rubric.py      # 8 Rubrics Definition
│   └── tools/         # External API Tools (Keyword, Spelling, Dictionary, Term, ETRI)
├── frontend/          # SPA (index.html, app.js, style.css)
├── data/              # Rubric criteria, prompts, score distributions, normative data
├── adapter/           # Large LoRA weights (Git LFS)
└── run.sh             # Execution script
```

## 📚 Data Source

- Scoring Model Training: [AI Hub Essay, Descriptive, and Thematic Writing Evaluation Data](https://aihub.or.kr). (Due to licensing, the raw data is not included in this repository).

## 🤝 Contributing

This project is open-source, and contributions are welcome! We appreciate bug reports, feature improvement ideas, and Pull Requests.

## 📄 License

This project is distributed under the MIT License. For more details, see the `LICENSE` file.
