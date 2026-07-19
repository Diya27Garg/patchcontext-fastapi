import os
import json
from typing import List, Dict, Any

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/chunks.json"

MAX_DISCUSSION_CHARS = 3000  # cap merged discussion text to keep chunks reasonably sized


def safe_text(value) -> str:
    """Return a stripped string, treating None as empty."""
    return (value or "").strip()


def load_json(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(RAW_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_chunk(text: str, source_type: str, source_number: int, url: str,
               chunk_kind: str, authors: List[str] = None, extra: Dict = None) -> Dict[str, Any]:
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


def merge_comments(comments: List[Dict], max_chars: int) -> str:
    """Merge all comments into a single truncated block of text."""
    parts = []
    total_len = 0
    for c in comments:
        author = c.get("author", "unknown")
        body = safe_text(c.get("body"))
        if not body:
            continue
        piece = f"[{author}]: {body}"
        if total_len + len(piece) > max_chars:
            break
        parts.append(piece)
        total_len += len(piece)
    return "\n\n".join(parts)


def chunk_issues(issues: List[Dict]) -> List[Dict]:
    chunks = []
    for issue in issues:
        number = issue["number"]
        url = issue["url"]

        body_text = safe_text(issue.get("body"))
        if body_text:
            chunks.append(make_chunk(
                text=f"Issue #{number}: {issue['title']}\n\n{body_text}",
                source_type="issue",
                source_number=number,
                url=url,
                chunk_kind="body",
                authors=[],
                extra={"title": issue["title"]}
            ))

        merged = merge_comments(issue.get("comments", []), MAX_DISCUSSION_CHARS)
        if merged:
            chunks.append(make_chunk(
                text=f"Discussion on Issue #{number} ({issue['title']}):\n\n{merged}",
                source_type="issue",
                source_number=number,
                url=url,
                chunk_kind="comments",
                authors=list({c.get("author", "unknown") for c in issue.get("comments", [])}),
                extra={"title": issue["title"]}
            ))
    return chunks


def chunk_pull_requests(prs: List[Dict]) -> List[Dict]:
    chunks = []
    for pr in prs:
        number = pr["number"]
        url = pr["url"]

        body_text = safe_text(pr.get("body"))
        if body_text:
            chunks.append(make_chunk(
                text=f"Pull Request #{number}: {pr['title']}\n\n{body_text}",
                source_type="pull_request",
                source_number=number,
                url=url,
                chunk_kind="body",
                authors=[],
                extra={"title": pr["title"], "merged": pr.get("merged", False)}
            ))

        # Merge discussion + review comments together into one chunk
        all_comments = pr.get("discussion_comments", []) + pr.get("review_comments", [])
        merged = merge_comments(all_comments, MAX_DISCUSSION_CHARS)
        if merged:
            chunks.append(make_chunk(
                text=f"Discussion on PR #{number} ({pr['title']}):\n\n{merged}",
                source_type="pull_request",
                source_number=number,
                url=url,
                chunk_kind="comments",
                authors=list({c.get("author", "unknown") for c in all_comments}),
                extra={"title": pr["title"], "merged": pr.get("merged", False)}
            ))
    return chunks


if __name__ == "__main__":
    print("Loading raw data...")
    issues = load_json("issues.json")
    prs = load_json("pull_requests.json")
    print(f"Loaded {len(issues)} issues, {len(prs)} PRs. (Skipping commits for this pass.)\n")

    print("Chunking (consolidated mode)...")
    all_chunks = []
    all_chunks.extend(chunk_issues(issues))
    all_chunks.extend(chunk_pull_requests(prs))

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    print(f"Created {len(all_chunks)} chunks total.\n")

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved chunks to {OUTPUT_PATH}")
