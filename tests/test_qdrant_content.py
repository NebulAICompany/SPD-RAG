import json
import os
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"
OUTPUT_FILE = "qdrant_metadata.json"

client = QdrantClient(url=QDRANT_URL)

all_metadata = []
offset = None
batch_size = 100

while True:
    results, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        offset=offset,
        limit=batch_size,
        with_payload=True,
        with_vectors=False,
    )

    for point in results:
        all_metadata.append({"id": point.id, "payload": point.payload})

    if next_offset is None:
        break
    offset = next_offset

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_metadata)} points to {OUTPUT_FILE}")
