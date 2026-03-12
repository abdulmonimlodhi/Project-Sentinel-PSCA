# Project Sentinel: PSCA Hybrid Multi-Agent System

Project Sentinel is a proof-of-concept AI orchestration framework designed for the Punjab Safe Cities Authority (PSCA). It simulates a highly reliable, automated Control Room environment using a hierarchical multi-agent system.

This project solves the "Agentic Churn" (infinite loops) and hallucination problems common in generative AI by implementing a **Hybrid Supervisor Pattern**, ensuring that law enforcement protocols are strictly followed while maintaining full auditability.

---

## Core Architecture: The Hybrid Supervisor

The Supervisor node is the central intelligence of the system. It uses a 3-layer decision-making process to guarantee deterministic reliability in high-stakes environments:

1. **The Law (Deterministic Rules):** Hardcoded state flags dictate the mandatory sequence of operations (e.g., Intelligence checks cannot run until the Observer has confirmed a visual match).
2. **The Counsel (LLM Reasoning):** The LLM reviews the emergency transcript and current state to suggest the next logical step.
3. **The Veto (Validation Layer):** The system cross-references the LLM's suggestion with the deterministic rules. If the LLM attempts to skip a vital protocol, the Supervisor overrides the LLM and enforces the rule.

---

## Advanced Features

* **Structured Data Passing (`detected_plate`):** Instead of forcing agents to parse long conversation histories using brittle regex, critical data (like license plates) is explicitly saved to the LangGraph `State`. This guarantees 100% accuracy when data is passed from the Observer to the Intelligence unit.
* **Role-Specific Prompt Engineering:** Each node utilizes system prompts tailored to specific law enforcement personas (e.g., Tactical Vision Specialist, DB Intelligence Analyst, Privacy Officer) to maintain a highly professional and context-accurate tone.
* **Zero-Temperature Redaction:** The Supervisor and Compliance nodes operate at `temperature=0` to eliminate "creative roleplay" and ensure strict adherence to PII (Personally Identifiable Information) masking rules.
* **Stateful Memory:** Uses LangGraph's `MemorySaver` to track incident threads over time, vital for ongoing investigations.
* **100% Auditability:** Every LLM prompt, deterministic override, and final output is traced via LangSmith, ensuring transparency for legal review.

---

## System Components (The Workforce)

The workflow is managed by a `StateGraph` consisting of specialized nodes:

| Agent / Node | Primary Responsibility | Simulated Action |
| --- | --- | --- |
| **Supervisor** | Orchestration and routing | Directs traffic between nodes and strictly enforces PSCA protocols. |
| **Observer** | Visual Analysis | Scans CCTV metadata to confirm vehicle presence, extracts license plates, and formats tactical alerts. |
| **Intelligence** | Database & OSINT | Reads plates from the shared State to query government databases for vehicle registration and active warrants. |
| **Compliance** | Privacy & Ethics | Acts as the final gatekeeper, masking PII and formatting the official dispatch report for public record. |

---

## Tech Stack

* **Orchestration:** LangGraph & LangChain
* **Language Model:** Groq (Llama-3.1-8b-instant)
* **Tracing & Auditing:** LangSmith
* **Environment Management:** Python & `python-dotenv`

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd ProjectSentinelPSCA

```

**2. Install dependencies**

```bash
pip install langchain-groq langchain-community langgraph python-dotenv langsmith

```

**3. Configure Environment Variables**
Create a `.env` file in the root directory:

```env
GROQ_API_KEY="your_groq_api_key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="PSCA-Sentinel-Hybrid"
LANGCHAIN_API_KEY="your_langsmith_api_key"

```

---

## Usage

### 1. Run the Real-Time Simulation

To execute a single emergency response workflow (e.g., a hit-and-run incident):

```bash
python sentinel_system.py

```

*Observe the terminal output as the Supervisor routes the incident through Observer, Intelligence, and Compliance before outputting a redacted final report.*

### 2. Run the Benchmark Suite

To test the system against multiple emergency scenarios and log the traces for auditing:

```bash
python benchmark_sentinel.py

```

*Once complete, log into your LangSmith dashboard to view the execution traces, latency, and Supervisor decision logs.*

---