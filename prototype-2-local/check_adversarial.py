import json

with open("data/eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    if d["id"] in (19, 20):
        print(f"Q{d['id']}: {d['question']}")
        print(d["answer"])
        print("-" * 60)