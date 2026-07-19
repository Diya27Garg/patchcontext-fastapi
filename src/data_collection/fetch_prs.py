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


def fetch_pull_requests(max_pages=15, per_page=30):
    """
    Fetch closed (merged or not) pull requests from the FastAPI repo.
    """
    all_prs = []

    for page in range(1, max_pages + 1):
        print(f"Fetching PRs page {page}...")
        url = f"{BASE_URL}/pulls"
        params = {
            "state": "closed",
            "per_page": per_page,
            "page": page,
            "sort": "popularity",   # prioritize PRs with more comments/reactions
            "direction": "desc"
        }

        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        if not data:
            print("No more PRs found.")
            break

        for item in data:
            all_prs.append({
                "number": item["number"],
                "title": item["title"],
                "body": item.get("body", ""),
                "state": item["state"],
                "merged": item.get("merged_at") is not None,
                "created_at": item["created_at"],
                "closed_at": item.get("closed_at"),
                "merged_at": item.get("merged_at"),
                "url": item["html_url"],
                "labels": [label["name"] for label in item.get("labels", [])],
                "base_branch": item["base"]["ref"],
            })

        time.sleep(0.5)

    return all_prs


def fetch_pr_review_comments(pr_number):
    """Fetch inline code review comments for a specific PR."""
    url = f"{BASE_URL}/pulls/{pr_number}/comments"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Error fetching review comments for PR #{pr_number}: {response.status_code}")
        return []

    comments = response.json()
    result = []
    for c in comments:
        user = c.get("user")
        result.append({
            "author": user["login"] if user else "unknown",
            "body": c.get("body", ""),
            "created_at": c.get("created_at"),
            "path": c.get("path"),
            "diff_hunk": c.get("diff_hunk")
        })
    return result


def fetch_pr_issue_comments(pr_number):
    """Fetch general discussion comments on a PR (uses the issues endpoint, since PRs are issues too)."""
    url = f"{BASE_URL}/issues/{pr_number}/comments"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Error fetching discussion comments for PR #{pr_number}: {response.status_code}")
        return []

    comments = response.json()
    result = []
    for c in comments:
        user = c.get("user")
        result.append({
            "author": user["login"] if user else "unknown",
            "body": c.get("body", ""),
            "created_at": c.get("created_at")
        })
    return result


if __name__ == "__main__":
    print("Starting FastAPI pull request collection...\n")

    prs = fetch_pull_requests(max_pages=15, per_page=30)
    print(f"\nFetched {len(prs)} pull requests.")

    print("\nFetching review comments and discussion for each PR...")
    for i, pr in enumerate(prs, 1):
        pr["review_comments"] = fetch_pr_review_comments(pr["number"])
        pr["discussion_comments"] = fetch_pr_issue_comments(pr["number"])
        time.sleep(0.3)

        if i % 20 == 0:
            print(f"  Processed {i}/{len(prs)} PRs...")
            # Save progress every 20 PRs in case of a crash
            os.makedirs("data/raw", exist_ok=True)
            with open("data/raw/pull_requests.json", "w", encoding="utf-8") as f:
                json.dump(prs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(prs)} pull requests to {output_path}")