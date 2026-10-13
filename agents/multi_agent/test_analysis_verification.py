"""Unit and regression tests for Analysis and Verification agents."""

import sys
from pathlib import Path
from unittest.mock import patch

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_local_dir = str(Path(__file__).resolve().parent)
if _local_dir in sys.path:
    sys.path.remove(_local_dir)

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.multi_agent.analysis_graph import analysis_graph
from agents.multi_agent.agents import analysis_agent, verification_agent
from agents.multi_agent.verification import (
    is_empty_search_result,
    has_unsupported_quality_claims,
    detect_contradiction,
    format_factual_analysis,
    verify_analysis_claims,
)


# 1. Test: Empty Jira search results
def test_empty_jira_search_analysis_and_verification():
    research_text = "No unresolved payment bugs were returned by the Jira query."

    # Verify helper detection
    assert is_empty_search_result(research_text) is True

    # Test analysis formatting for empty search
    raw_analysis = "No payment bugs were found."
    formatted = format_factual_analysis(research_text, raw_analysis)
    assert "outside the search scope" in formatted
    assert not has_unsupported_quality_claims(formatted)

    # Test verification
    is_valid, verified_msg = verify_analysis_claims(research_text, formatted)
    assert is_valid is True
    assert "does not make unsupported claims about overall system quality" in verified_msg

    # Test full verification_agent node with mock LLM
    mock_llm_response = AIMessage(
        content="The analysis accurately reflects the search result and does not make unsupported claims about overall system quality."
    )
    from langchain_ollama import ChatOllama
    with patch.object(ChatOllama, "invoke", return_value=mock_llm_response):
        state = {
            "user_request": "Find unresolved payment bugs",
            "research_result": research_text,
            "analysis_result": formatted,
            "verification_result": "",
            "next_agent": "",
            "final_response": "",
        }
        res = verification_agent(state)
        assert "does not make unsupported claims about overall system quality" in res["verification_result"]
        assert "does not make unsupported claims about overall system quality" in res["final_response"]


# 2. Test: Research results containing open and resolved issues
def test_research_containing_open_and_resolved_issues():
    research_text = (
        "Here are the payment-related bugs found:\n"
        "1. BUG-101: Payment timeout - High priority, open issue.\n"
        "2. BUG-102: Payment callback failure - High priority, resolved issue.\n"
        "3. BUG-103: Payment retry duplicate - Critical priority, open issue."
    )

    assert is_empty_search_result(research_text) is False

    analysis_text = (
        "Found 3 issues in total. Open issues: BUG-101 (High) and BUG-103 (Critical). "
        "Resolved issues: BUG-102 (High)."
    )

    formatted = format_factual_analysis(research_text, analysis_text)
    assert "BUG-101" in formatted
    assert "BUG-102" in formatted
    assert "BUG-103" in formatted

    is_valid, verified_msg = verify_analysis_claims(research_text, formatted)
    assert is_valid is True
    assert "accurately reflects the issues" in verified_msg
    assert not has_unsupported_quality_claims(verified_msg)


# 3. Test: Unsupported quality claims are detected and removed
def test_unsupported_quality_claims_detection_and_removal():
    research_text = "No unresolved payment bugs were found in Jira based on your search query."

    flawed_analysis = (
        "The Jira system has been thoroughly checked, and no unresolved payment bugs were identified. "
        "This indicates a strong level of quality assurance and maintenance in the system. "
        "The system has high quality and high stability, meeting the highest standards of quality."
    )

    # 1. Detection
    assert has_unsupported_quality_claims(flawed_analysis) is True

    # 2. Sanitization in format_factual_analysis
    sanitized = format_factual_analysis(research_text, flawed_analysis)
    assert has_unsupported_quality_claims(sanitized) is False
    assert "outside the search scope" in sanitized

    # 3. Verification must flag and remove unsupported quality claims
    is_valid, verified_msg = verify_analysis_claims(research_text, flawed_analysis)
    assert is_valid is False
    assert "Unsupported claim removed" in verified_msg
    assert "does not establish overall system quality" in verified_msg

    # Full agent node verification
    mock_llm_response = AIMessage(content="The system has high quality.")
    from langchain_ollama import ChatOllama
    with patch.object(ChatOllama, "invoke", return_value=mock_llm_response):
        state = {
            "user_request": "Find unresolved payment bugs",
            "research_result": research_text,
            "analysis_result": flawed_analysis,
            "verification_result": "",
            "next_agent": "",
            "final_response": "",
        }
        res = verification_agent(state)
        assert "Unsupported claim removed" in res["verification_result"]
        assert has_unsupported_quality_claims(res["final_response"]) is False


# 4. Test: Analysis claims that contradict the research
def test_analysis_claims_contradicting_research():
    research_text = "No unresolved payment bugs were returned by the Jira query."
    contradictory_analysis = "Three critical payment bugs were found in production (BUG-101, BUG-103)."

    # Contradiction detection
    contradiction = detect_contradiction(research_text, contradictory_analysis)
    assert contradiction is not None
    assert "claims issues exist, but Research found no matching issues" in contradiction

    # Verification must flag contradiction
    is_valid, verified_msg = verify_analysis_claims(research_text, contradictory_analysis)
    assert is_valid is False
    assert "Contradiction detected" in verified_msg

    # Reverse contradiction: Research has issues, but analysis says none exist
    research_with_issues = "Found issue SAM1-7: Create Wallet Integration (In Progress)."
    contradictory_analysis_empty = "No matching issues were found in Jira."
    rev_contradiction = detect_contradiction(research_with_issues, contradictory_analysis_empty)
    assert rev_contradiction is not None
    assert "claims no issues exist, but Research identified matching issues" in rev_contradiction

    is_valid2, verified_msg2 = verify_analysis_claims(research_with_issues, contradictory_analysis_empty)
    assert is_valid2 is False
    assert "Contradiction detected" in verified_msg2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
