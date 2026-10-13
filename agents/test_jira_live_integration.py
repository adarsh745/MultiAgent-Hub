"""Live integration test for Jira Cloud issue retrieval using local .env.

This script executes real, read-only requests against the configured Atlassian Jira Cloud instance.
Never exposes or prints the API token.
"""

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.services.jira_service import JiraService
from agents.multi_agent.tools import search_jira_issues, build_jql_from_query


def run_live_integration_test():
    print("=" * 60)
    print("Jira Cloud Live Integration Test")
    print("=" * 60)

    # 1. Configuration Check
    service = JiraService()
    is_valid, error = service.validate_config()
    if not is_valid:
        print(f"[FAIL] Configuration error: {error}")
        return False

    print(f"[INFO] Jira Base URL: {service.base_url}")
    print(f"[INFO] Jira Email:    {service.email}")
    print(f"[INFO] API Token:     [CONFIGURED AND MASKED]")

    # 2. Real Read-Only Search: Project issues
    print("\n--- Test 1: Real Issue Retrieval from Accessible Projects ---")
    jql_all = "project in (KAN, SAM1) ORDER BY updated DESC"
    result = service.search_issues_sync(jql_all, max_results=5)
    issues = result.get("issues", [])
    total = result.get("total", 0)

    print(f"Retrieved {total} real issues from Jira Cloud.")
    assert len(issues) > 0, "Expected at least 1 real issue in projects KAN or SAM1."

    sample = issues[0]
    print(f"Sample Issue:")
    print(f"  - Key:      {sample['key']}")
    print(f"  - Summary:  {sample['summary']}")
    print(f"  - Status:   {sample['status']}")
    print(f"  - Priority: {sample['priority']}")
    print(f"  - Assignee: {sample['assignee']}")
    print(f"  - Created:  {sample['created']}")
    print(f"  - Updated:  {sample['updated']}")

    # Verify required keys are normalized
    for req_key in ["key", "summary", "status", "priority", "assignee", "created", "updated"]:
        assert req_key in sample, f"Missing normalized field '{req_key}'"
    print("[PASS] Test 1: Real issue retrieval and normalization verified.")

    # 3. Real Read-Only Search: Query with 0 matches (Payment)
    print("\n--- Test 2: Honest Reporting for Empty Results (Payment Bugs) ---")
    jql_payment = build_jql_from_query("Find unresolved payment bugs in Jira")
    print(f"Built JQL: {jql_payment}")
    res_payment = service.search_issues_sync(jql_payment, max_results=10)
    payment_issues = res_payment.get("issues", [])
    print(f"Payment issues found: {len(payment_issues)}")
    assert len(payment_issues) == 0, "Expected 0 payment issues in user's Jira instance."
    print("[PASS] Test 2: Empty results handled honestly without errors or fake data.")

    # 4. LangChain Tool Invocation: Real Search for 'wallet'
    print("\n--- Test 3: LangChain Tool Invocation (Real Search for 'wallet') ---")
    wallet_tool_result = search_jira_issues.invoke({"query": "wallet"})
    print("Tool Output:")
    print(wallet_tool_result[:300] + "...")
    assert "SAM1-7" in wallet_tool_result or "Wallet" in wallet_tool_result
    parsed_wallet = json.loads(wallet_tool_result)
    assert isinstance(parsed_wallet, list) and len(parsed_wallet) >= 1
    print("[PASS] Test 3: search_jira_issues returned real Jira issue data.")

    # 5. LangChain Tool Invocation: 'payment' query
    print("\n--- Test 4: LangChain Tool Invocation ('Find unresolved payment bugs') ---")
    payment_tool_result = search_jira_issues.invoke({"query": "Find unresolved payment bugs in Jira"})
    print("Tool Output:")
    print(payment_tool_result)
    assert "No matching Jira issues found" in payment_tool_result
    print("[PASS] Test 4: search_jira_issues correctly reported no matching issues.")

    print("\n" + "=" * 60)
    print("[SUCCESS] All Live Jira Cloud Integration Tests Passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_live_integration_test()
    sys.exit(0 if success else 1)
