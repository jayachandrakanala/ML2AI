from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from agents.state import AgentState
from agents.summarizer import summarizer_agent
from agents.quiz_generator import quiz_agent
from agents.reviewer import reviewer_agent


def human_quiz_node(state: AgentState) -> Dict[str, Any]:
    """
    NATIVE HUMAN-IN-THE-LOOP NODE:
    Pauses graph execution using interrupt() and waits for the student's quiz answers.
    Execution halts here until Command(resume=...) is passed from Streamlit.
    """
    logs = state.get("activity_logs", [])
    logs.append({
        "agent": "Human-in-the-Loop",
        "action": "⏸️ Graph paused via interrupt(). Waiting for student quiz submission..."
    })
    
    # Interrupt execution and pause graph thread
    user_answers = interrupt({
        "message": "Please complete the quiz assessment in Tab 2.",
        "quiz_questions": state.get("quiz_questions", [])
    })
    
    # Execution resumes here when Command(resume=user_answers) is invoked
    return {
        "user_answers": user_answers,
        "activity_logs": logs
    }


def create_study_assistant_graph(llm):
    workflow = StateGraph(AgentState)
    
    # 1. Add nodes
    workflow.add_node("summarizer", lambda state: summarizer_agent(state, llm))
    workflow.add_node("quiz_generator", lambda state: quiz_agent(state, llm))
    workflow.add_node("human_quiz", human_quiz_node)
    workflow.add_node("reviewer", lambda state: reviewer_agent(state, llm))
    
    # 2. Define sequence
    workflow.set_entry_point("summarizer")
    workflow.add_edge("summarizer", "quiz_generator")
    workflow.add_edge("quiz_generator", "human_quiz")
    workflow.add_edge("human_quiz", "reviewer")
    
    # 3. Conditional Self-Correction / Terminal Check
    def check_review_status(state: AgentState):
        # If user submitted answers and reviewer finished evaluation, complete graph
        if state.get("user_answers"):
            return END
        # If reviewer passed pre-validation without corrections, complete initial run
        if state.get("reviewer_passed", True):
            return END
        # Otherwise, loop back to regenerate/fix questions
        return "quiz_generator"
            
    workflow.add_conditional_edges(
        "reviewer",
        check_review_status,
        {
            "quiz_generator": "quiz_generator",
            END: END
        }
    )
    
    # 4. Compile with MemorySaver Checkpointer (REQUIRED for interrupt/Command support)
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)