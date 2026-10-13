"""Standalone connection test for Slack Web API integration.

Verifies bot identity, team/workspace, and configuration using the root .env.
Never prints, logs, or exposes the SLACK_BOT_TOKEN.
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

slack_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
slack_default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "").strip()
slack_api_url = os.getenv("SLACK_API_BASE_URL", "https://slack.com/api").rstrip("/")


def test_slack_connection() -> bool:
    """Execute Slack auth.test API call and display connection diagnostics."""
    print("=" * 60)
    print("Slack Web API Connection Test")
    print("=" * 60)
    print(f"Loaded configuration from: {dotenv_path if dotenv_path else 'default environment'}")
    print(f"Slack Default Channel:     {slack_default_channel or '[NOT SET / OPTIONAL]'}")
    print(f"Slack Bot Token:           {'[CONFIGURED AND MASKED]' if slack_token else '[NOT SET]'}")

    if not slack_token:
        print("\n[ERROR] Missing required SLACK_BOT_TOKEN in .env.")
        print("Please configure SLACK_BOT_TOKEN with your Slack bot user OAuth token (xoxb-...).")
        return False

    url = f"{slack_api_url}/auth.test"
    print(f"\nConnecting to {url}...")

    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "WorkPilot-AI",
    }

    try:
        response = httpx.post(url, headers=headers, timeout=15.0)
        status = response.status_code

        try:
            data = response.json()
        except Exception:
            data = {}

        is_ok = data.get("ok", False)

        if status == 200 and is_ok:
            team_name = data.get("team", "Unknown Team")
            team_id = data.get("team_id", "Unknown Team ID")
            user_name = data.get("user", "Unknown User")
            user_id = data.get("user_id", "Unknown User ID")
            bot_id = data.get("bot_id", "Unknown Bot ID")
            workspace_url = data.get("url", "Unknown URL")

            print("\n[SUCCESS] Slack API connection verified successfully!")
            print("-" * 60)
            print(f"Workspace / Team: {team_name} ({team_id})")
            print(f"Workspace URL:    {workspace_url}")
            print(f"Bot Identity:     @{user_name} ({user_id})")
            print(f"Bot ID:           {bot_id}")
            print("-" * 60)
            return True

        # Error cases
        error_code = data.get("error", "unknown_error")
        print(f"\n[FAILURE] Slack API request failed with HTTP {status} (ok={is_ok})")
        print(f"Slack Error Code: {error_code}")

        if status == 429 or error_code == "ratelimited":
            retry_after = response.headers.get("Retry-After", "unknown")
            print(f"[DIAGNOSIS] Slack rate limit exceeded. Retry after: {retry_after}s.")
        elif status == 401 or error_code in ("invalid_auth", "not_authed", "account_inactive", "token_revoked", "token_expired"):
            print("[DIAGNOSIS] Authentication failed. Check that SLACK_BOT_TOKEN starts with 'xoxb-' and is active in your Slack workspace.")
        elif error_code in ("missing_scope", "not_allowed_token_type"):
            needed = data.get("needed", "unknown")
            provided = data.get("provided", "unknown")
            print(f"[DIAGNOSIS] Permission error. Missing required scope: '{needed}'. Currently provided scopes: '{provided}'.")
        elif status >= 500:
            print("[DIAGNOSIS] Slack internal server error. Check Slack System Status (https://status.slack.com).")
        else:
            print(f"[DIAGNOSIS] Unexpected error returned by Slack: {error_code}")

        return False

    except httpx.TimeoutException:
        print("\n[FAILURE] Network timeout while attempting to connect to Slack API.")
        print("[DIAGNOSIS] Check your internet connection or proxy settings.")
        return False
    except httpx.RequestError as exc:
        print(f"\n[FAILURE] Network connectivity error: {exc}")
        return False
    except Exception as exc:
        print(f"\n[FAILURE] Unexpected exception: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    success = test_slack_connection()
    sys.exit(0 if success else 1)
