import os
import warnings
from typing import TypedDict, Annotated, List, Dict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# --- 1. MONITORING & TRACING (LANGSMITH) ---
# Ensure these variables are in your .env file for automatic tracing
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "PSCA-Sentinel-Hybrid"

# Suppress minor deprecation warnings for a clean console output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 2. STATE DEFINITION ---
class SentinelState(TypedDict):
    """
    Maintains the state of the PSCA operation.
    Using Annotated[List, add_messages] allows history to persist across nodes.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str
    vision_checked: bool
    intel_checked: bool
    compliance_approved: bool

# --- 3. SPECIALIZED NODES (The Workforce) ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)

def observer_node(state: SentinelState):
    """
    Node 1: Computer Vision / CCTV Analysis
    Simulates a Vision Model output and uses the LLM to format a formal report.
    """
    print("\n[Node: Observer] Processing CCTV Metadata...")

    # SIMULATION: In reality, this data would come from a YOLO or CLIP model
    simulated_vision_data = {
        "timestamp": "2026-03-12 12:45:00",
        "camera_id": "LRT-CAM-04",
        "detection": "Silver SUV",
        "confidence": 0.98,
        "license_plate": "LHR-892",
        "trajectory": "Heading North toward Main Blvd"
    }

    # We use the LLM to turn raw metadata into a professional alert
    obs_prompt = f"""You are the Vision Analysis Agent. 
    Summarize this raw metadata into a concise police alert: {simulated_vision_data}"""
    
    response = llm.invoke([SystemMessage(content=obs_prompt)])
    
    return {
        "messages": [AIMessage(content=response.content, name="Observer")],
        "vision_checked": True
    }

def intelligence_node(state: SentinelState):
    """
    Node 2: Intelligence & OSINT
    Extracts the plate from history and simulates a Database Query.
    """
    print("\n[Node: Intelligence] Accessing Excise & Criminal Records...")

    # 1. LOGIC: Extract the plate from the last message (the Observer's report)
    # Chain data between agents
    last_msg = state["messages"][-1].content
    # Simple logic: assume the plate is LHR-892 for the demo if not found
    plate_found = "LHR-892" if "LHR-892" in last_msg else "UNKNOWN"

    # SIMULATION: Database lookup based on the plate found
    db_record = {
        "plate": plate_found,
        "owner": "Arshad Ali",
        "status": "Active Registration",
        "warrants": "None",
        "theft_report": "Negative",
        "last_known_address": "Sector G-10, Islamabad"
    }

    # 2. LLM: Summarize the database findings
    intel_prompt = f"You are the Intelligence Agent. Summarize this database record for the dispatch team: {db_record}"
    response = llm.invoke([SystemMessage(content=intel_prompt)])

    return {
        "messages": [AIMessage(content=response.content, name="Intelligence")],
        "intel_checked": True
    }

def compliance_node(state: SentinelState):
    """
    Node 3: Ethics & PII Compliance
    Strictly redacts PII and formats the final report.
    """
    print("\n[Node: Compliance] Redacting Sensitive Data & Generating Report...")

    # We use a dedicated prompt that forces a specific structure
    compliance_prompt = """
    You are a PSCA Privacy & Compliance Officer. 
    Your ONLY task is to summarize the incident for the public record while protecting PII.

    STRICT REDACTION RULES:
    1. License Plates: Must be masked (e.g., LHR-***).
    2. Names: Must be redacted (e.g., A*** Ali or [REDACTED]).
    3. Do NOT invent new dialogue or events. Only use facts from the history.

    REQUIRED OUTPUT FORMAT:
    ### PSCA INCIDENT SUMMARY REPORT
    - **Incident Type**: [Type]
    - **Location**: [Location]
    - **Vehicle Info**: [Color/Type]
    - **Plate Status**: [Masked Plate]
    - **Owner Status**: [Redacted Name / Record Status]
    - **Action Taken**: [Final recommendation based on history]
    """

    # We inject the compliance prompt as the final instruction (HumanMessage)
    # to overcome the "recency bias" of the police narrative.
    messages = state["messages"] + [HumanMessage(content=compliance_prompt)]
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content, name="Compliance")],
        "compliance_approved": True
    }

# --- 4. THE SUPERVISOR (Orchestration Layer) ---
def supervisor_node(state: SentinelState) -> Dict:
    """
    Supervisor agent that orchestrates the workflow.
    Logic Flow: 
    1. Deterministic rules (Primary Logic)
    2. LLM suggestion (Secondary Guidance)
    3. Validation to ensure rules are respected
    """
    print("\n[Supervisor] Orchestrating Response...")
    
    # --- LAYER 1: DETERMINISTIC RULES (Source of Truth) ---
    v = state.get("vision_checked", False)
    i = state.get("intel_checked", False)
    c = state.get("compliance_approved", False)

    if not v:
        forced_next = "Observer"
    elif not i:
        forced_next = "Intelligence"
    elif not c:
        forced_next = "Compliance"
    else:
        forced_next = "FINISH"

    # --- LAYER 2: LLM REASONING (The 'Consultant') ---
    prompt = f"""Review the conversation history above. Your ONLY job is to route the incident to the correct next specialized unit.

    CURRENT PROTOCOL STATUS:
    - Vision Checked: {v}
    - Intelligence Checked: {i}
    - Compliance Approved: {c}

    ROUTING RULES:
    1. If Vision Checked is False -> Output: Observer
    2. If Vision Checked is True but Intelligence is False -> Output: Intelligence
    3. If Vision and Intelligence are True but Compliance is False -> Output: Compliance
    4. If all are True -> Output: FINISH

    Respond with EXACTLY ONE WORD from this list: [Observer, Intelligence, Compliance, FINISH].
    Do not add any explanation, reasoning, or conversational text."""
    
    # FIX: Put the history first, and the strict instruction LAST as a HumanMessage
    messages_for_llm = state["messages"] + [HumanMessage(content=prompt)]
    
    llm_suggestion = llm.invoke(messages_for_llm).content.strip()
    
    # Strip out any random punctuation the LLM might still try to add
    llm_suggestion = llm_suggestion.replace(".", "").replace("\n", "").strip()
    
    print(f"LLM Suggestion: {llm_suggestion}")

    # --- LAYER 3: VALIDATION & OVERRIDE ---
    # We ensure the LLM cannot skip steps required by the Safe City protocols
    if forced_next.lower() in llm_suggestion.lower():
        final_target = forced_next
        supervisor_msg = f"Supervisor: Proceeding with protocol-aligned unit: {final_target}."
    else:
        print(f"!!! Protocol Violation Detected: LLM suggested {llm_suggestion}, but rules require {forced_next} !!!")
        final_target = forced_next
        supervisor_msg = f"Supervisor: Overriding suggestion to maintain protocol. Assigning to {final_target}."

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_node": final_target
    }

# --- 5. GRAPH CONSTRUCTION ---

workflow = StateGraph(SentinelState)

# Add all nodes to the graph
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Observer", observer_node)
workflow.add_node("Intelligence", intelligence_node)
workflow.add_node("Compliance", compliance_node)

# Flow logic: Always return to Supervisor after a task to check the next step
workflow.add_edge(START, "Supervisor")
workflow.add_edge("Observer", "Supervisor")
workflow.add_edge("Intelligence", "Supervisor")
workflow.add_edge("Compliance", "Supervisor")

# Final router logic
def router_logic(state: SentinelState):
    if state["next_node"] == "FINISH":
        return END
    return state["next_node"]

workflow.add_conditional_edges("Supervisor", router_logic)

# Compile with memory to allow thread persistence
app = workflow.compile(checkpointer=MemorySaver())

# --- 6. RUNTIME SIMULATION ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "PSCA_HYBRID_DEMO_01"}}
    
    incident_call = "15 Emergency: A hit-and-run occurred at Liberty Roundabout. A silver SUV is fleeing toward Main Blvd."
    
    # Explicitly initialize state flags
    initial_input = {
        "messages": [HumanMessage(content=incident_call)],
        "vision_checked": False,
        "intel_checked": False,
        "compliance_approved": False
    }

    print("--- PSCA PROJECT SENTINEL: HYBRID SUPERVISOR MODE ---")
    for event in app.stream(initial_input, config=config, stream_mode="values"):
        if event["messages"]:
            event["messages"][-1].pretty_print()