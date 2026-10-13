"""Live integration test for Slack API integration using local .env credentials.

Tests real Slack Web API endpoints:
1. Authentication verification (auth.test)
2. Channel listing (conversations.list)
3. Channel access & history check (conversations.history)
4. LangChain tool invocation (search_slack_messages)

Never prints or logs the SLACK_BOT_TOKEN.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root and backend are on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = _project_root / ".env"
load_dotenv(dotenv_path)

from app.services.slack_service import (
    SlackService,
    SlackServiceError,
    SlackNotInChannelError,
    SlackPermissionError,
)
from agents.multi_agent.tools import search_slack_messages


def run_live_tests():
    print("=" * 60)
    print("Slack Web API Live Integration Test")
    print("=" * 60)

    service = SlackService()
    is_valid, error = service.validate_config()

    print(f"[INFO] Slack Default Channel: {service.default_channel or '[NOT SET / OPTIONAL]'}")
    print(f"[INFO] Slack Bot Token:       {'[CONFIGURED AND MASKED]' if service._token else '[NOT SET]'}")

    if not is_valid:
        print(f"[SKIP] Slack credentials not configured: {error}")
        return True

    # -------------------------------------------------------------
    # Test 1: Real Authentication (auth.test)
    # -------------------------------------------------------------
    print("\n--- Test 1: Real Authentication Verification ---")
    try:
        auth_data = service.auth_test_sync()
        assert auth_data["ok"] is True
        print(f"Team / Workspace: {auth_data.get('team')} ({auth_data.get('team_id')})")
        print(f"Bot User:         @{auth_data.get('user')} ({auth_data.get('user_id')})")
        print(f"Bot ID:           {auth_data.get('bot_id')}")
        print("[PASS] Test 1: Real authentication verified.")
    except Exception as exc:
        print(f"[FAIL] Test 1 Failed: {exc}")
        return False

    # -------------------------------------------------------------
    # Test 2: Real Public Channel Listing (conversations.list)
    # -------------------------------------------------------------
    print("\n--- Test 2: Public Channel Listing ---")
    try:
        channels = service.list_channels_sync(types="public_channel")
        print(f"Retrieved {len(channels)} public channel(s):")
        for ch in channels:
            print(f"  - #{ch['name']} (ID: {ch['id']}, is_member: {ch['is_member']}, members: {ch['num_members']})")
        assert isinstance(channels, list)
        print("[PASS] Test 2: Real public channels retrieved.")
    except Exception as exc:
        print(f"[FAIL] Test 2 Failed: {exc}")
        return False

    # -------------------------------------------------------------
    # Test 3: Channel History or Membership Status Check
    # -------------------------------------------------------------
    print("\n--- Test 3: Channel History & Membership Verification ---")
    member_channels = [ch for ch in channels if ch["is_member"]]
    if member_channels:
        target = member_channels[0]
        try:
            messages = service.get_channel_history_sync(target["id"], limit=5)
            print(f"Retrieved {len(messages)} message(s) from member channel #{target['name']}.")
            print("[PASS] Test 3: Channel history retrieved successfully.")
        except Exception as exc:
            print(f"[FAIL] Test 3 Failed: {exc}")
            return False
    else:
        # Check that conversations.history properly raises SlackNotInChannelError for non-member channel
        non_member = channels[0] if channels else None
        if non_member:
            try:
                service.get_channel_history_sync(non_member["id"], limit=5)
                print("[UNEXPECTED] Expected not_in_channel error, but succeeded.")
            except SlackNotInChannelError:
                print(f"Confirmed expected behavior: bot is not in #{non_member['name']} and Slack returned not_in_channel.")
                print("[PASS] Test 3: Channel permission boundary correctly enforced.")
            except Exception as exc:
                print(f"[FAIL] Test 3 Failed: {exc}")
                return False
        else:
            print("[INFO] No public channels in workspace to test.")
            print("[PASS] Test 3: Skipped (no channels).")

    # -------------------------------------------------------------
    # Test 4: LangChain Tool Invocation (search_slack_messages)
    # -------------------------------------------------------------
    print("\n--- Test 4: LangChain search_slack_messages Tool ---")
    try:
        tool_output = search_slack_messages.invoke({"query": "payment discussion"})
        print("Tool Output (first 300 chars):")
        print(tool_output[:300] + ("..." if len(tool_output) > 300 else ""))
        assert tool_output is not None
        assert "Error: Slack is not configured" not in tool_output
        print("[PASS] Test 4: search_slack_messages executed cleanly.")
    except Exception as exc:
        print(f"[FAIL] Test 4 Failed: {exc}")
        return False

    print("\n" + "=" * 60)
    print("[SUCCESS] All Live Slack Web API Integration Tests Passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_live_tests()
    sys.exit(0 if success else 1)
