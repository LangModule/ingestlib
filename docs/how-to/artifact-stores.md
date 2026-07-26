# Choose an artifact store

Artifacts — parses, page renders, figure crops, stage outputs — live on one
of two backends. Same layout, same API, one config key.

```yaml
artifact_store: s3      # or: local
```

## `local` — a plain folder, zero cloud

```yaml
artifact_store: local
# artifacts:
#   path: artifacts     # relative paths anchor beside config.yaml
```

Everything is ordinary files you can browse in a file manager:

```text
artifacts/documents/{doc_id}/parse/pages/page_0001.png
```

Writes are atomic (temp file + rename), so a crash mid-write never leaves
a truncated artifact. Moving a corpus to another machine — or to S3 later
— is a plain copy.

Best for: development, single-machine deployments, air-gapped setups.

## `s3` — durable and shareable

```yaml
artifact_store: s3
aws:
  profile: your-aws-profile
  region: us-east-1
  account_id: "123456789012"
# s3:
#   bucket: ingestlib-{account_id}    # the default
```

The bucket is created automatically on first use. Two S3 realities the
errors will walk you through if you hit them:

- **Bucket names are global across all AWS accounts** — if your chosen
  name is taken by someone else, the error says so; pick a unique
  `s3.bucket`.
- **A typo'd `aws.profile` fails loudly** listing your actual available
  profiles — it never silently falls back to default credentials.

Best for: teams sharing a corpus, production, anything multi-machine.

## One API over both

Your code never branches on the backend:

```python
from ingestlib.storage import artifacts

artifacts.list_documents()
artifacts.load_parse(doc_id)
artifacts.read_blob(artifacts.page_image_key(doc_id, 1))   # PNG bytes either way
artifacts.delete_document(doc_id)
```

## Why artifacts matter

The artifact store is the **source of truth**; the vector store is a
rebuildable index over it. Because parses persist here, re-ingesting into
a different vector store — or after an embedding-provider switch — reuses
every stored parse and repeats zero OCR/LLM work.

---

Next: [Switch AI providers](switch-providers.md).
