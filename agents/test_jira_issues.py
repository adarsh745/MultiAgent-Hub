
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
email = os.getenv("JIRA_EMAIL")
api_token = os.getenv("JIRA_API_TOKEN")

if not all([base_url, email, api_token]):
    raise RuntimeError("Missing Jira configuration in .env")

url = f"{base_url}/rest/api/3/project/search"

try:
    response = httpx.get(
        url,
        auth=(email, api_token),
        headers={"Accept": "application/json"},
        params={"maxResults": 10},
        timeout=15.0,
    )

    response.raise_for_status()
    data = response.json()

    projects = data.get("values", [])
    print(f"Projects found: {len(projects)}")

    for project in projects:
        print(
            f"Key: {project.get('key')} | "
            f"Name: {project.get('name')}"
        )

    if not projects:
        print("No accessible projects found.")

except httpx.HTTPStatusError as exc:
    print("Jira API returned:", exc.response.status_code)
    print(exc.response.text[:500])
except httpx.RequestError as exc:
    print("Connection error:", type(exc).__name__)
