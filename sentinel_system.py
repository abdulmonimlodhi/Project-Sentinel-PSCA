import os
import warnings
from typing import TypedDict, Annotated, List, Dict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# --- 1. MONITORING & TRACING ---
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "PSCA-Sentinel-Hybrid"

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 2. STATE DEFINITION ---
class SentinelState(TypedDict):
    """
    Maintains the state of the PSCA operation.
    Added 'detected_plate' for reliable data passing between agents.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str
    vision_checked: bool
    intel_checked: bool
    compliance_approved: bool
    detected_plate: str  # <--- NEW: Explicit data slot for the plate

# --- 3. SPECIALIZED NODES ---
# Setting temperature to 0 for the Supervisor and Compliance for deterministic results
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def observer_node(state: SentinelState):
    """
    Node 1: Computer Vision Analysis.
    Uses LLM to transform raw CCTV metadata into a tactical police alert.
    """
    print("\n[Node: Observer] Processing CCTV Metadata...")

    sim_data = {
        "detection": "Silver SUV",
        "license_plate": "LHR-892",
        "direction": "North toward Main Blvd",
        "confidence": "98%"
    }

    # Prompt forces a tactical, police-radio style response
    obs_prompt = f"""
    You are a PSCA Vision Specialist. 
    Transform this raw CCTV detection into a professional, high-priority police alert.
    DATA: {sim_data}
    
    KEEP IT CONCISE. Use bullet points for the vehicle specs.
    """
    
    response = llm.invoke([SystemMessage(content=obs_prompt)])
    
    return {
        "messages": [AIMessage(content=response.content, name="Observer")],
        "vision_checked": True,
        "detected_plate": sim_data["license_plate"]
    }

def intelligence_node(state: SentinelState):
    """
    Node 2: Intelligence & OSINT.
    Reads the plate from state and formats a confidential background report.
    """
    print("\n[Node: Intelligence] Querying Government Databases...")

    plate = state.get("detected_plate", "UNKNOWN")
    db_record = {
        "plate": plate,
        "owner": "Arshad Ali",
        "status": "Active Registration",
        "criminal_history": "None",
        "theft_report": "Negative"
    }

    # Prompt focuses on 'Database Reporting' tone
    intel_prompt = f"""
    You are a PSCA Intelligence Analyst. 
    Summarize these database findings into a 'Confidential Intelligence Report'.
    FINDINGS: {db_record}
    
    Ensure you clearly state if the vehicle is stolen or if the owner has warrants.
    """
    
    response = llm.invoke([SystemMessage(content=intel_prompt)])

    return {
        "messages": [AIMessage(content=response.content, name="Intelligence")],
        "intel_checked": True
    }

def compliance_node(state: SentinelState):
    """
    Node 3: Ethics & PII Compliance.
    The final 'Filter' that ensures the public record is safe to share.
    """
    print("\n[Node: Compliance] Redacting Sensitive Data & Generating Report...")

    # Strict template enforcement ensures no roleplay hallucinations
    compliance_prompt = """
    
    TEMPLATE:
    You are a PSCA Privacy & Compliance Officer. 
    Your ONLY task is to summarize the incident for the public record while protecting PII.

    STRICT REDACTION RULES:
    1. License Plates: Must be masked (e.g., LHR-***).
    2. Names: Must be redacted (e.g., A*** Ali or [REDACTED]).
    3. Do NOT invent new dialogue or events. Only use facts from the history.

    REQUIRED OUTPUT FORMAT:
    ### PSCA INCIDENT SUMMARY REPORT
    - **Incident**: [Brief description]
    - **Location**: [Location]
    - **Vehicle Info**: [Color/Type]
    - **Plate Status**: [Masked Plate]
    - **Owner Status**: [Redacted Name / Record Status]
    - **Final Action**: [Dispatch/Close]
    """

    messages = state["messages"] + [HumanMessage(content=compliance_prompt)]
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content, name="Compliance")],
        "compliance_approved": True
    }

# --- 4. THE SUPERVISOR ---
def supervisor_node(state: SentinelState) -> Dict:
    print("\n[Supervisor] Orchestrating Response...")
    
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
    
    messages_for_llm = state["messages"] + [HumanMessage(content=prompt)]
    llm_suggestion = llm.invoke(messages_for_llm).content.strip().replace(".", "")
    
    print(f"LLM Suggestion: {llm_suggestion}")

    if forced_next.lower() in llm_suggestion.lower():
        final_target = forced_next
        supervisor_msg = f"Supervisor: Proceeding with protocol-aligned unit: {final_target}."
    else:
        print(f"!!! Protocol Violation Override: LLM suggested {llm_suggestion} !!!")
        final_target = forced_next
        supervisor_msg = f"Supervisor: Overriding to maintain protocol. Assigning to {final_target}."

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_node": final_target
    }

# --- 5. GRAPH CONSTRUCTION ---
workflow = StateGraph(SentinelState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Observer", observer_node)
workflow.add_node("Intelligence", intelligence_node)
workflow.add_node("Compliance", compliance_node)

workflow.add_edge(START, "Supervisor")
workflow.add_edge("Observer", "Supervisor")
workflow.add_edge("Intelligence", "Supervisor")
workflow.add_edge("Compliance", "Supervisor")

def router_logic(state: SentinelState):
    if state["next_node"] == "FINISH":
        return END
    return state["next_node"]

workflow.add_conditional_edges("Supervisor", router_logic)
app = workflow.compile(checkpointer=MemorySaver())

# --- 6. RUNTIME SIMULATION ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "PSCA_FINAL_DEMO"}}
    
    initial_input = {
        "messages": [HumanMessage(content="15 Emergency: Hit-and-run at Liberty Roundabout. Silver SUV fleeing toward Main Blvd.")],
        "vision_checked": False,
        "intel_checked": False,
        "compliance_approved": False,
        "detected_plate": "" # Initialized empty
    }

    print("--- PSCA PROJECT SENTINEL: HYBRID SUPERVISOR MODE ---")
    for event in app.stream(initial_input, config=config, stream_mode="values"):
        if event["messages"]:
            event["messages"][-1].pretty_print()