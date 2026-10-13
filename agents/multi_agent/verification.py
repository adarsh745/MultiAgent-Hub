"""Verification and claim-validation logic for Analysis and Verification agents."""

import re
from typing import Optional, Tuple

# Patterns identifying unsupported extrapolations about system quality or health
UNSUPPORTED_QUALITY_PATTERNS = [
    r"\bhigh quality\b",
    r"\bstrong level of quality\b",
    r"\bhighest standards?\b",
    r"\bhigh stability\b",
    r"\bstate of high quality\b",
    r"\bflawless\b",
    r"\bperfect state\b",
    r"\bcompletely free of (?:all )?bugs\b",
    r"\bquality assurance and maintenance\b",
    r"\bno further action is (?:required|needed)\b",
    r"\bmeets the highest standards\b",
    r"\bstrong quality assurance\b",
]

# Patterns identifying empty search results
EMPTY_SEARCH_PATTERNS = [
    r"\bno (?:matching )?(?:jira )?issues (?:were )?found\b",
    r"\bno (?:unresolved )?(?:[a-zA-Z_-]+ )?bugs (?:were )?found\b",
    r"\bno (?:matching )?unresolved (?:[a-zA-Z_-]+ )?bugs were returned\b",
    r"\b0 (?:matching )?issues\b",
    r"\bno issues found\b",
    r"\bno issues match\b",
    r"\bno (?:matching )?(?:pull requests?|prs?|commits?) (?:were )?found\b",
    r"\bno (?:matching )?(?:slack )?messages (?:were )?found\b",
    r"\bno results (?:found|returned)\b",
]



def is_empty_search_result(text: str) -> bool:
    """Return True if research text indicates an empty search result."""
    lower = text.lower()
    return any(re.search(p, lower) for p in EMPTY_SEARCH_PATTERNS)


def has_unsupported_quality_claims(text: str) -> bool:
    """Return True if text makes unsupported claims regarding system quality or stability."""
    lower = text.lower()
    return any(re.search(p, lower) for p in UNSUPPORTED_QUALITY_PATTERNS)


def detect_contradiction(research: str, analysis: str) -> Optional[str]:
    """
    Detect if the analysis directly contradicts the research findings.
    Returns a description of the contradiction, or None if consistent.
    """
    research_is_empty = is_empty_search_result(research)
    analysis_is_empty = is_empty_search_result(analysis)

    # Contradiction 1: Research found 0 issues, but analysis asserts issues exist or were resolved
    if research_is_empty:
        # Check if analysis claims specific bugs were found, discussed, or resolved
        has_bug_id = bool(re.search(r"\b(?:BUG|SAM1|KAN|PAY)-\d+\b", analysis, re.I))
        claims_open_count = bool(re.search(r"\b(?:[1-9]\d*)\s+(?:open|unresolved|critical|high priority)\s+(?:issues|bugs)\b", analysis, re.I))
        claims_issues_exist = "issues were found" in analysis.lower() or "bugs were found" in analysis.lower()
        claims_issues_resolved = any(p in analysis.lower() for p in [
            "closed the bug", "resolved the issue", "resolved the payment issue",
            "discussed in slack", "fixed the bug", "fixed the issue",
        ])
        if (has_bug_id or claims_open_count or claims_issues_exist or claims_issues_resolved) and not analysis_is_empty:
            return "Analysis claims issues exist or were resolved/discussed, but Research found no matching issues or messages."


    # Contradiction 2: Research found issues, but analysis asserts no issues exist
    if not research_is_empty:
        research_has_issues = bool(re.search(r"\b(?:BUG|SAM1|KAN)-\d+\b", research, re.I)) or "key" in research.lower()
        if research_has_issues and analysis_is_empty:
            return "Analysis claims no issues exist, but Research identified matching issues."

    return None


def format_factual_analysis(research: str, raw_analysis: str) -> str:
    """
    Ensure the analysis is concise, factual, and does not infer unsupported quality claims.
    Distinguishes between no matches returned vs. absence across entire project.
    """
    # If research is an empty search result
    if is_empty_search_result(research):
        # Strip any unsupported quality claims
        cleaned = raw_analysis
        for pattern in UNSUPPORTED_QUALITY_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.I)

        # Remove repetitive/excessive boilerplate sections
        if has_unsupported_quality_claims(raw_analysis) or len(cleaned.splitlines()) > 6:
            # Extract target topic if mentioned (e.g. payment bugs)
            match_topic = re.search(r"no (?:unresolved )?([a-zA-Z_-]+(?: [a-zA-Z_-]+)?) (?:bugs|issues)", research, re.I)
            topic = match_topic.group(1) if match_topic else "unresolved"
            return (
                f"The query returned no matching {topic} bugs. "
                "This does not establish that no such bugs exist outside the search scope."
            )

        # Ensure scope limitation disclaimer is present
        if "outside the search scope" not in cleaned and "limited to the search" not in cleaned:
            cleaned = cleaned.rstrip(".") + ". This does not establish that no such bugs exist outside the search scope."

        return cleaned.strip()

    # If research contains matching issues, remove any quality claims while preserving factual details
    cleaned = raw_analysis
    for pattern in UNSUPPORTED_QUALITY_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I)

    # Clean up double spaces or awkward punctuation from removals
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned.strip()


def verify_analysis_claims(
    research: str,
    analysis: str,
    llm_conclusion: str = "",
) -> Tuple[bool, str]:
    """
    Verify each claim in the analysis against the research findings.
    
    Returns:
        (is_valid, verified_conclusion):
            - is_valid: True if analysis is supported and free of unsupported quality claims.
            - verified_conclusion: Concise, factual verified text.
    """
    contradiction = detect_contradiction(research, analysis)
    if contradiction:
        return False, (
            f"Contradiction detected: {contradiction} "
            "Corrected finding: The search result strictly governs the findings."
        )

    has_quality_claim = has_unsupported_quality_claims(analysis)
    research_is_empty = is_empty_search_result(research)

    if has_quality_claim:
        return False, (
            "Unsupported claim removed: A search returning no matching issues does not "
            "establish overall system quality or stability. The verified finding is limited "
            "to the search scope performed."
        )

    if research_is_empty:
        return True, (
            "The analysis accurately reflects the search result and does not make "
            "unsupported claims about overall system quality."
        )

    # For non-empty research results
    if llm_conclusion and not has_unsupported_quality_claims(llm_conclusion):
        return True, llm_conclusion.strip()

    return True, (
        "The analysis accurately reflects the issues identified in the research result "
        "without unsupported claims."
    )
