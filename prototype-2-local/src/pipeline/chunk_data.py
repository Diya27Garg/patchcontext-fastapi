import os
import json
from typing import List, Dict, Any

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/chunks.json"

WINDOW_SIZE = 4
WINDOW_OVERLAP = 1


def safe_text(value) -> str:
    return (value or "").strip()


def load_json(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(RAW_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_chunk(text, source_type, source_number, url, chunk_kind, authors=None, extra=None):
    chunk = {
        "text": text.strip(),
        "source_type": source_type,
        "source_number": source_number,
        "url": url,
        "chunk_kind": chunk_kind,
        "authors": authors or [],
    }
    if extra:
        chunk.update(extra)
    return chunk


def window_comments(comments, window_size, overlap):
    if not comments:
        return []
    step = max(window_size - overlap, 1)
    windows = []
    for start in range(0, len(comments), step):
        window = comments[start:start + window_size]
        if window:
            windows.append(window)
        if start + window_size >= len(comments):
            break
    return windows


def format_comment_window(window):
    parts = []
    for c in window:
        author = c.get("author", "unknown")
        body = safe_text(c.get("body"))
        if body:
            parts.append(f"[{author}]: {body}")
    return "\n\n".join(parts)


def chunk_issues(issues):
    chunks = []
    for issue in issues:
        number = issue["number"]
        url = issue["url"]

        body_text = safe_text(issue.get("body"))
        if body_text:
            chunks.append(make_chunk(
                text=f"Issue #{number}: {issue['title']}\n\n{body_text}",
                source_type="issue", source_number=number, url=url,
                chunk_kind="body", extra={"title": issue["title"]}
            ))

        for w in window_comments(issue.get("comments", []), WINDOW_SIZE, WINDOW_OVERLAP):
            text = format_comment_window(w)
            if text.strip():
                chunks.append(make_chunk(
                    text=f"Discussion on Issue #{number} ({issue['title']}):\n\n{text}",
                    source_type="issue", source_number=number, url=url,
                    chunk_kind="comments",
                    authors=list({c.get("author", "unknown") for c in w}),
                    extra={"title": issue["title"]}
                ))
    return chunks


def chunk_pull_requests(prs):
    chunks = []
    for pr in prs:
        number = pr["number"]
        url = pr["url"]

        body_text = safe_text(pr.get("body"))
        if body_text:
            chunks.append(make_chunk(
                text=f"Pull Request #{number}: {pr['title']}\n\n{body_text}",
                source_type="pull_request", source_number=number, url=url,
                chunk_kind="body", extra={"title": pr["title"], "merged": pr.get("merged", False)}
            ))

        for w in window_comments(pr.get("discussion_comments", []), WINDOW_SIZE, WINDOW_OVERLAP):
            text = format_comment_window(w)
            if text.strip():
                chunks.append(make_chunk(
                    text=f"Discussion on PR #{number} ({pr['title']}):\n\n{text}",
                    source_type="pull_request", source_number=number, url=url,
                    chunk_kind="comments",
                    authors=list({c.get("author", "unknown") for c in w}),
                    extra={"title": pr["title"], "merged": pr.get("merged", False)}
                ))

        for w in window_comments(pr.get("review_comments", []), WINDOW_SIZE, WINDOW_OVERLAP):
            text = format_comment_window(w)
            if text.strip():
                chunks.append(make_chunk(
                    text=f"Code review discussion on PR #{number} ({pr['title']}):\n\n{text}",
                    source_type="pull_request", source_number=number, url=url,
                    chunk_kind="review_comments",
                    authors=list({c.get("author", "unknown") for c in w}),
                    extra={"title": pr["title"], "merged": pr.get("merged", False)}
                ))
    return chunks


def chunk_commits(commits):
    chunks = []
    for commit in commits:
        message = safe_text(commit.get("message"))
        if not message:
            continue
        chunks.append(make_chunk(
            text=f"Commit {commit['sha'][:7]} by {commit.get('author_name', 'unknown')}:\n\n{message}",
            source_type="commit", source_number=None, url=commit["url"],
            chunk_kind="body", authors=[commit.get("author_login", "unknown")],
            extra={"sha": commit["sha"], "date": commit.get("date")}
        ))
    return chunks


if __name__ == "__main__":
    print("Loading raw data...")
    issues = load_json("issues.json")
    prs = load_json("pull_requests.json")
    commits = load_json("commits.json")
    print(f"Loaded {len(issues)} issues, {len(prs)} PRs, {len(commits)} commits.\n")

    print("Chunking (full quality, all sources, sliding windows)...")
    all_chunks = []
    all_chunks.extend(chunk_issues(issues))
    all_chunks.extend(chunk_pull_requests(prs))
    all_chunks.extend(chunk_commits(commits))

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    print(f"Created {len(all_chunks)} chunks total.\n")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved chunks to {OUTPUT_PATH}")
