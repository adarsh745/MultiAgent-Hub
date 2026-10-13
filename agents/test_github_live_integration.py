"""Live integration test for GitHub API integration using local .env.

Executes real, read-only requests against GitHub API for adarsh745/Production-ready-RAG-.
Never prints or exposes the GITHUB_TOKEN.
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

from app.services.github_service import GitHubService
from agents.multi_agent.tools import search_github_repository


def run_live_integration_test():
    print("=" * 60)
    print("GitHub API Live Integration Test")
    print("=" * 60)

    # 1. Configuration check
    service = GitHubService()
    is_valid, error = service.validate_config()
    if not is_valid:
        print(f"[FAIL] Configuration error: {error}")
        return False

    print(f"[INFO] GitHub Owner: {service.owner}")
    print(f"[INFO] GitHub Repo:  {service.repo}")
    print(f"[INFO] GitHub Token: [CONFIGURED AND MASKED]")

    # 2. Test 1: Repository metadata
    print("\n--- Test 1: Real Repository Metadata Retrieval ---")
    repo_meta = service.get_repository_sync()
    print(f"Repository Full Name: {repo_meta['full_name']}")
    print(f"Default Branch:       {repo_meta['default_branch']}")
    print(f"Visibility:           {repo_meta['visibility']}")
    assert repo_meta["full_name"] == f"{service.owner}/{service.repo}"
    assert repo_meta["default_branch"] is not None
    print("[PASS] Test 1: Real repository metadata verified.")

    # 3. Test 2: Real Commits Retrieval
    print("\n--- Test 2: Real Commit History Retrieval ---")
    commits = service.list_commits_sync(per_page=5)
    print(f"Retrieved {len(commits)} commits.")
    assert len(commits) > 0, "Expected at least 1 commit in repository."
    sample_commit = commits[0]
    print(f"Latest Commit:")
    print(f"  - SHA:     {sample_commit['sha']}")
    print(f"  - Author:  {sample_commit['author']}")
    print(f"  - Message: {sample_commit['message']}")
    print(f"  - Date:    {sample_commit['date']}")
    for k in ["sha", "full_sha", "message", "author", "date", "html_url"]:
        assert k in sample_commit, f"Missing normalized field '{k}'"
    print("[PASS] Test 2: Real commits verified.")

    # 4. Test 3: Pull Requests (honest reporting for 0 PRs)
    print("\n--- Test 3: Pull Requests Retrieval (0 PRs handled honestly) ---")
    prs = service.list_pull_requests_sync(state="all")
    print(f"Pull requests found: {len(prs)}")
    assert len(prs) == 0, "Expected 0 PRs in user repository."
    print("[PASS] Test 3: Empty pull requests handled honestly without fake data.")

    # 5. Test 4: LangChain Tool Invocation (commits)
    print("\n--- Test 4: search_github_repository Tool (Commits) ---")
    commits_result = search_github_repository.invoke({"query": "show recent commits in my repository"})
    data = json.loads(commits_result)
    assert data["resource"] == "commits"
    assert data["count"] > 0
    print(f"Tool returned {data['count']} commits. Top message: {data['items'][0]['message']}")
    print("[PASS] Test 4: Tool returned real GitHub commits data.")

    # 6. Test 5: LangChain Tool Invocation (pull requests)
    print("\n--- Test 5: search_github_repository Tool (Pull Requests) ---")
    prs_result = search_github_repository.invoke({"query": "Find recent pull requests in my GitHub repository"})
    print(f"Tool response: {prs_result}")
    assert "No matching pull requests found in GitHub repository" in prs_result
    print("[PASS] Test 5: Tool honestly reported 0 matching pull requests.")

    print("\n" + "=" * 60)
    print("[SUCCESS] All Live GitHub API Integration Tests Passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_live_integration_test()
    sys.exit(0 if success else 1)
