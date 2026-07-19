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


def fetch_commits(max_pages=20, per_page=30):
    """
    Fetch commit history from the FastAPI repo (main/master branch).
    """
    all_commits = []

    for page in range(1, max_pages + 1):
        print(f"Fetching commits page {page}...")
        url = f"{BASE_URL}/commits"
        params = {
            "per_page": per_page,
            "page": page,
        }

        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        if not data:
            print("No more commits found.")
            break

        for item in data:
            commit_info = item.get("commit", {})
            author_info = commit_info.get("author", {})
            gh_author = item.get("author")  # GitHub user object, can be None

            all_commits.append({
                "sha": item["sha"],
                "message": commit_info.get("message", ""),
                "author_name": author_info.get("name", "unknown"),
                "author_login": gh_author["login"] if gh_author else "unknown",
                "date": author_info.get("date"),
                "url": item["html_url"],
            })

        time.sleep(0.5)

    return all_commits


if __name__ == "__main__":
    print("Starting FastAPI commit collection...\n")

    commits = fetch_commits(max_pages=20, per_page=30)  # ~600 commits
    print(f"\nFetched {len(commits)} commits.")

    # Save to disk
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/commits.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(commits, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(commits)} commits to {output_path}")