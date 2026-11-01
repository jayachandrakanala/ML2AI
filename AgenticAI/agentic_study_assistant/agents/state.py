from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    # Inputs
    document_text: str
    user_topic: str
    user_answers: Optional[Dict[int, str]]  # e.g., {1: "A", 2: "C", ...}
    
    # Intermediaries & Outputs
    summary: str
    quiz_questions: List[Dict[str, Any]]
    
    # Reviewer Validation
    reviewer_passed: bool
    reviewer_feedback: str
    corrections_made: List[str]
    
    # Activity Logging
    activity_logs: List[Dict[str, str]]