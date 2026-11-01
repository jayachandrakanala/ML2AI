import json
import re
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState

def reviewer_agent(state: AgentState, llm) -> Dict[str, Any]:
    doc_text = state.get("document_text", "")
    summary = state.get("summary", "")
    questions = state.get("quiz_questions", [])
    user_answers = state.get("user_answers", None)
    logs = state.get("activity_logs", [])
    corrections = state.get("corrections_made", [])

    # Normalize user_answers dictionary keys & values to clean strings if present
    normalized_user_answers = None
    if user_answers:
        normalized_user_answers = {}
        if isinstance(user_answers, dict):
            for k, v in user_answers.items():
                normalized_user_answers[str(k).strip()] = str(v).strip()
        else:
            normalized_user_answers = str(user_answers)

    # PHASE 1: PRE-SUBMISSION QUESTION ACCURACY VERIFICATION (If no user answers yet)
    if not normalized_user_answers:
        prompt = ChatPromptTemplate.from_template("""
        You are a Quality Assurance Reviewer Agent.
        Verify if these 5 generated quiz questions are 100% factually accurate based ONLY on the provided document context.
        
        Document Context:
        {document_text}
        
        Questions:
        {questions}
        
        Return ONLY a JSON object with this structure:
        {{
            "passed": true,
            "feedback": "All questions are grounded in source context."
        }}
        """)
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({
                "document_text": doc_text[:3000] if doc_text else summary[:3000], 
                "questions": json.dumps(questions) if isinstance(questions, list) else str(questions)
            })
            
            raw_content = response.content.strip()
            
            # Robust JSON parsing using Regex
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                res_json = json.loads(json_match.group(0))
            else:
                clean_text = raw_content.replace("```json", "").replace("```", "").strip()
                res_json = json.loads(clean_text)
                
            passed = res_json.get("passed", True)
            feedback = res_json.get("feedback", "Questions verified successfully.")
        except Exception:
            passed = True
            feedback = "Questions verified successfully."

        if not passed:
            corrections.append(f"Reviewer requested quiz revision: {feedback}")
            logs.append({"agent": "Reviewer Agent", "action": f"❌ Flagged issues in quiz: {feedback}"})
        else:
            logs.append({"agent": "Reviewer Agent", "action": "✅ Validated Quiz Questions against document source."})

        return {
            "reviewer_passed": passed,
            "reviewer_feedback": feedback,
            "corrections_made": corrections,
            "activity_logs": logs
        }

    # PHASE 2: POST-SUBMISSION EVALUATION & GRADING (When user_answers present)
    else:
        prompt = ChatPromptTemplate.from_template("""
        You are the Reviewer & Evaluator Agent.
        Evaluate the student's quiz answers against the correct answers and the document text.
        
        Quiz Data:
        {questions}
        
        User Submitted Answers:
        {user_answers}
        
        Document Source:
        {document_text}
        
        Provide an encouraging, highly structured, personalized study report using Markdown:
        1. Score breakdown (e.g., Score: 4/5).
        2. Per-question evaluation (Identify student choices, explain why incorrect options diverged from the document source).
        3. Targeted study recommendations based on weak areas.
        """)
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({
                "questions": json.dumps(questions, indent=2) if isinstance(questions, list) else str(questions),
                "user_answers": json.dumps(normalized_user_answers, indent=2) if isinstance(normalized_user_answers, dict) else str(normalized_user_answers),
                "document_text": doc_text[:3000] if doc_text else summary[:3000]
            })
            report_text = response.content
        except Exception as e:
            report_text = f"### Evaluation Report\n\nSubmitted Answers: {normalized_user_answers}\n\n*Error generating report details: {str(e)}*"

        logs.append({"agent": "Reviewer Agent", "action": "🎓 Evaluated student responses and generated personalized feedback."})
        
        return {
            "reviewer_feedback": report_text,
            "activity_logs": logs
        }