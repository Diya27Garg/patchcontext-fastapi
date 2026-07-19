import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

REPO_OWNER = "tiangolo"
REPO_NAME = "fastapi"
BASE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"


def fetch_issues(max_pages=5, per_page=30):
    """
    Fetch closed issues (excluding PRs) from the FastAPI repo.
    GitHub's issues endpoint actually returns PRs too, so we filter those out.
    """
    all_issues = []

    for page in range(1, max_pages + 1):
        print(f"Fetching issues page {page}...")
        url = f"{BASE_URL}/issues"
        params = {
            "state": "closed",       # closed issues tend to have resolved discussions
            "per_page": per_page,
            "page": page,
            "sort": "comments",      # prioritize issues with more discussion
            "direction": "desc"
        }

        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        if not data:
            print("No more issues found.")
            break

        for item in data:
            # Skip pull requests (GitHub's issues API includes PRs too)
            if "pull_request" in item:
                continue

            all_issues.append({
                "number": item["number"],
                "title": item["title"],
                "body": item.get("body", ""),
                "state": item["state"],
                "created_at": item["created_at"],
                "closed_at": item.get("closed_at"),
                "comments_count": item["comments"],
                "url": item["html_url"],
                "labels": [label["name"] for label in item.get("labels", [])]
            })

        time.sleep(0.5)  # be polite to the API

    return all_issues


def fetch_issue_comments(issue_number):
    """Fetch all comments for a specific issue."""
    url = f"{BASE_URL}/issues/{issue_number}/comments"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Error fetching comments for issue #{issue_number}: {response.status_code}")
        return []

    comments = response.json()
    return [
        {
            "author": c["user"]["login"],
            "body": c["body"],
            "created_at": c["created_at"]
        }
        for c in comments
    ]


if __name__ == "__main__":
    print("Starting FastAPI issue collection...\n")

    issues = fetch_issues(max_pages=20, per_page=30)  # start small: ~90 issues to test
    print(f"\nFetched {len(issues)} issues (excluding PRs).")

    # Now fetch comments for each issue that actually has discussion
    print("\nFetching comments for issues with discussion...")
    for issue in issues:
        if issue["comments_count"] > 0:
            issue["comments"] = fetch_issue_comments(issue["number"])
        else:
            issue["comments"] = []
        time.sleep(0.3)

    # Save to disk
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/issues.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(issues)} issues with comments to {output_path}")