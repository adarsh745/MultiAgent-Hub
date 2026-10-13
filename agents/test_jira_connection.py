
import os
import sys
from pathlib import Path

import httpx
from dotenv import find_dotenv, load_dotenv

# Search for .env starting from current working directory or fall back to project root
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
email = os.getenv("JIRA_EMAIL", "").strip()
api_token = os.getenv("JIRA_API_TOKEN", "").strip()


def test_jira_connection() -> bool:
    print(f"Loaded configuration from: {dotenv_path if dotenv_path else 'default environment'}")
    print(f"Jira Base URL:  {base_url or '[NOT SET]'}")
    print(f"Jira Email:     {email or '[NOT SET]'}")
    print(f"Jira API Token: {'[CONFIGURED]' if api_token else '[NOT SET]'}")

    missing = []
    if not base_url:
        missing.append("JIRA_BASE_URL")
    if not email:
        missing.append("JIRA_EMAIL")
    if not api_token:
        missing.append("JIRA_API_TOKEN")

    if missing:
        print(f"\n[ERROR] Missing required environment variables in .env: {', '.join(missing)}")
        return False

    url = f"{base_url}/rest/api/3/myself"
    print(f"\nTesting endpoint: {url}")

    try:
        response = httpx.get(
            url,
            auth=(email, api_token),
            headers={"Accept": "application/json"},
            timeout=15.0,
        )

        status = response.status_code
        if status == 200:
            user = response.json()
            display_name = user.get("displayName", "N/A")
            account_id = user.get("accountId", "N/A")
            email_address = user.get("emailAddress", email)
            active = user.get("active", True)
            print("\n[SUCCESS] Jira connection established successfully!")
            print(f"  - Display Name: {display_name}")
            print(f"  - Account ID:   {account_id}")
            print(f"  - Email:        {email_address}")
            print(f"  - Active:       {active}")
            return True

        elif status == 401:
            print("\n[ERROR 401] Unauthorized: Authentication failed.")
            print("  - Please verify that JIRA_EMAIL matches your Atlassian account.")
            print("  - Ensure JIRA_API_TOKEN is a valid API token generated from https://id.atlassian.com/manage-profile/security/api-tokens.")

        elif status == 403:
            print("\n[ERROR 403] Forbidden: Access denied.")
            print("  - Your account may not have permission to access Jira Software on this site.")
            print("  - Check site-level permissions or product access in Atlassian administration.")

        elif status == 404:
            print("\n[ERROR 404] Not Found: Site or endpoint unavailable.")
            print(f"  - Verify JIRA_BASE_URL ({base_url}).")
            print("  - If this is a newly created site, confirm the subdomain is fully provisioned.")
            print(f"  - Atlassian response: {response.text[:300]}")

        elif status == 429:
            print("\n[ERROR 429] Rate Limit Exceeded: Too many requests sent to the Jira API.")
            print("  - Please wait before retrying.")

        elif 500 <= status < 600:
            print(f"\n[ERROR {status}] Atlassian Server Error:")
            print("  - The Jira Cloud service is experiencing internal issues.")
            print("  - Check Atlassian status page: https://status.atlassian.com")
            print(f"  - Response body: {response.text[:300]}")

        else:
            print(f"\n[ERROR {status}] Unexpected response from Jira:")
            print(f"  - Response: {response.text[:300]}")

        return False

    except httpx.TimeoutException:
        print("\n[ERROR] Connection timed out while connecting to Jira.")
        print(f"  - Check network connectivity and whether {base_url} is reachable.")
        return False
    except httpx.ConnectError as exc:
        print(f"\n[ERROR] Network connection error: Failed to connect to {base_url}.")
        print(f"  - Details: {exc}")
        return False
    except httpx.RequestError as exc:
        print(f"\n[ERROR] Network request error: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    success = test_jira_connection()
    sys.exit(0 if success else 1)
