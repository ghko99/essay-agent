# Geulgyeol (Automated Essay Scoring Agent)

![License](https://img.shields.io/github/license/ghko99/essay-agent)
![Stars](https://img.shields.io/github/stars/ghko99/essay-agent?style=social)
![Issues](https://img.shields.io/github/issues/ghko99/essay-agent)

**Demo**: https://geulgyeol.tech

<img width="1182" height="941" alt="Screenshot" src="https://github.com/user-attachments/assets/7204bae3-17f9-4733-a417-0dc6a5607cf3" />

## Project Introduction

**Geulgyeol** (Korean for "Texture of Text") is an open-source automated scoring service that evaluates student essays based on 8 specific rubrics. Rather than just assigning a score, an AI agent utilizes real-world tools—such as a spell checker, dictionary, and language analysis API—to measure the text and present a comprehensive report explaining **why** the score was given with objective evidence.

### Scoring Rubrics (8 Categories)
Each rubric is scored from 1 to 9. Detailed criteria for each level can be found on the **Scoring Criteria** page of the demo site.

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

<img width="1240" height="940" alt="Screenshot" src="https://github.com/user-attachments/assets/ba8ae3f0-f8b5-4e8c-8a33-96363b323d8e" />

## Reason for Making

Evaluating essays manually is a time-consuming and repetitive task. Existing generative AI models also suffer from hallucinations and the **anchoring problem**—where a verification model simply copies the original score without conducting real evaluation. 

We started this project to automate the scoring process robustly. By concealing the base score from the verification agent and forcing it to evaluate essays independently based strictly on tool measurements, we created a system where users can receive highly reliable, evidence-based feedback without the repetitive manual work.

## Main Features

### 1. 3-Step Scoring Pipeline
```text
Essay Input
   │
   ▼
① Ensemble Scoring (LoRA Fine-tuned Model)
   └─ 8 Rubrics × 1~9 Points, Weighted Self-Consistency Ensemble
   │
   ▼
② Agent Verification (Base Model + External Tools)
   └─ Independently measures text via tools → Evidence-based scoring → Score calibration (±3)
   │
   ▼
③ Final Report
   └─ Overall Review · Strengths · Areas for Improvement · Rubric Scores and Evidence
```
<img width="1240" height="943" alt="Screenshot" src="https://github.com/user-attachments/assets/7783ca8f-6a47-47f4-8496-5d397ad635aa" />

### 2. External APIs (Tools) for Independent Verification
| Tool | Purpose |
|---|---|
| Keyword Fulfillment | Measures if the prompt's key terms are present |
| Spell Checker (Bareun) | Detects spelling and spacing errors |
| Korean Dictionary | Verifies if the vocabulary consists of real words |
| Tech Term Verification | Checks the accurate use of technical terms |
| ETRI Language Analysis | Morpheme, dependency parsing, and semantic roles |
| Perplexity Measurement | Measures sentence naturalness (custom model) |

### 3. Tech Stack
- **Scoring Model**: kanana-1.5-8b-instruct-2505 + LoRA Adapter
- **Inference Server**: vLLM (AsyncLLMEngine)
- **Backend**: FastAPI (Fully async, SSE Streaming)
- **Frontend**: Vanilla JS SPA

## How to Use

### Getting Started
1. Check the example data and rubric guidelines on the demo site.
2. Install dependencies (Requires Python 3.10 and CUDA GPU):
   ```bash
   pip install -r requirements.txt
   ```
3. Configure External API Keys (Create a `.env` file):
   `OPENDICT_API_KEY`, `KRD_API_KEY`, `KTERM_API_KEY`, `ETRI_API_KEY`, `BAREUN_API_KEY`
4. Run the Server:
   ```bash
   KANANA_BASE=/path/to/kanana ./run.sh
   # → http://localhost:8000
   ```
   *(Note: The LoRA adapter is managed with Git LFS. Run `git lfs pull` after cloning.)*

### Project Structure
```text
essay_agent/
├── backend/
│   ├── main.py        # FastAPI endpoints (Scoring, Verification, Report SSE)
│   ├── model.py       # vLLM Engine, Ensemble Scoring, LoRA Toggle
│   ├── agent.py       # Verification Agent (Tool Loop, Independent Scoring, Report)
│   ├── rubric.py      # 8 Rubrics Definition
│   └── tools/         # External API Tools
├── frontend/          # SPA (index.html, app.js, style.css)
├── data/              # Rubric criteria, prompts, normative data
├── adapter/           # Large LoRA weights (Git LFS)
└── run.sh             # Execution script
```

## Future Plans

- **Usage Examples**: Add more prompt test cases and sample essays.
- **Multilingual Support**: Integrate OpenAI APIs to expand the evaluation pipeline to support multiple languages (e.g., English).
- **Automation Improvements**: Refine the multi-step reasoning capabilities of the agent for better, nuanced feedback.
- **Robust Error Handling**: Improve exception handling during external tool execution to prevent interruptions in the agent loop.
