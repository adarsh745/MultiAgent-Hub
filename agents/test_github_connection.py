"""Standalone connection test for GitHub API integration.

Verifies repository access and configuration using the root .env.
Never prints or exposes the GITHUB_TOKEN.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import find_dotenv, load_dotenv

# Search for .env starting from current directory up to project root
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

github_token = os.getenv("GITHUB_TOKEN", "").strip()
github_owner = os.getenv("GITHUB_OWNER", "").strip()
github_repo = os.getenv("GITHUB_REPO", "").strip()
github_api_url = os.getenv("GITHUB_API_BASE_URL", "https://api.github.com").rstrip("/")


def test_github_connection() -> bool:
    print(f"Loaded configuration from: {dotenv_path if dotenv_path else 'default environment'}")
    print(f"GitHub Owner:  {github_owner or '[NOT SET]'}")
    print(f"GitHub Repo:   {github_repo or '[NOT SET]'}")
    print(f"GitHub Token:  {'[CONFIGURED]' if github_token else '[NOT SET]'}")

    missing = []
    if not github_token:
        missing.append("GITHUB_TOKEN")
    if not github_owner:
        missing.append("GITHUB_OWNER")
    if not github_repo:
        missing.append("GITHUB_REPO")

    if missing:
        print(f"\n[ERROR] Missing required environment variables in .env: {', '.join(missing)}")
        return False

    url = f"{github_api_url}/repos/{github_owner}/{github_repo}"
    print(f"\nTesting endpoint: {url}")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "WorkPilot-AI",
    }

    try:
        response = httpx.get(url, headers=headers, timeout=15.0)
        status = response.status_code

        if status == 200:
            data = response.json()
            full_name = data.get("full_name", f"{github_owner}/{github_repo}")
            visibility = data.get("visibility", "N/A")
            default_branch = data.get("default_branch", "N/A")
            description = data.get("description") or "(No description provided)"

            print("\n[SUCCESS] GitHub repository connection established successfully!")
            print(f"  - Full Name:      {full_name}")
            print(f"  - Visibility:     {visibility}")
            print(f"  - Default Branch: {default_branch}")
            print(f"  - Description:    {description}")
            return True

        elif status == 401:
            print("\n[ERROR 401] Unauthorized: GitHub authentication failed.")
            print("  - Please verify that GITHUB_TOKEN is a valid GitHub personal access token.")
            print("  - Check token expiration and ensure it has not been revoked.")

        elif status == 403:
            remaining = response.headers.get("x-ratelimit-remaining", "unknown")
            print(f"\n[ERROR 403] Forbidden: Access denied or rate limited (remaining quota: {remaining}).")
            print("  - Ensure your personal access token has permission to access this repository.")
            print("  - If this is a fine-grained token, verify resource owner and repository access permissions.")

        elif status == 404:
            print("\n[ERROR 404] Not Found: Repository not found.")
            print(f"  - Verify GITHUB_OWNER ('{github_owner}') and GITHUB_REPO ('{github_repo}').")
            print("  - For private repositories, ensure GITHUB_TOKEN has permission to view the repository.")

        elif status == 429:
            print("\n[ERROR 429] Rate Limit Exceeded: Too many requests sent to GitHub API.")
            print("  - Please wait before retrying.")

        elif 500 <= status < 600:
            print(f"\n[ERROR {status}] GitHub Server Error:")
            print("  - GitHub is experiencing server-side issues. Check https://www.githubstatus.com.")

        else:
            print(f"\n[ERROR {status}] Unexpected response from GitHub:")
            print(f"  - Details: {response.text[:300]}")

        return False

    except httpx.TimeoutException:
        print("\n[ERROR] Connection timed out while contacting GitHub API.")
        print(f"  - Verify network connectivity to {github_api_url}.")
        return False
    except httpx.ConnectError as exc:
        print(f"\n[ERROR] Network connection error: Failed to connect to {github_api_url}.")
        print(f"  - Details: {exc}")
        return False
    except httpx.RequestError as exc:
        print(f"\n[ERROR] Network request error: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    success = test_github_connection()
    sys.exit(0 if success else 1)
