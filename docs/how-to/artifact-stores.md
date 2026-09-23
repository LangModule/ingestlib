# Choose an artifact store

Artifacts — parses, page renders, figure crops, stage outputs — live on `s3`
(AWS, or a self-hosted MinIO) or a plain `local` folder. Same layout, same API,
one config key.

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

A least-privilege IAM policy for the default stack (Bedrock + the
artifact bucket + Amazon Rerank) lives in the repo's
[`infra/` folder](https://github.com/LangModule/ingestlib/tree/main/infra)
— replace the placeholders and attach.

### Self-hosted S3 (MinIO)

The same `s3` backend can point at a self-hosted, S3-compatible store like
[MinIO](https://min.io) instead of AWS — no cloud account, and the bytes never
leave your box. Set an endpoint and use static keys instead of an AWS profile:

```yaml
artifact_store: s3
s3:
  bucket: ingestlib
  endpoint_url: http://localhost:9000    # your MinIO
```

```bash
# .env — MinIO access keys (no aws.profile needed when an endpoint is set)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

Bring up a MinIO with the bundled compose:
`docker compose -f infra/docker-compose.yml --profile minio up -d` (S3 API on
9000, web console on 9001). This makes the whole S3 path — including
`ingestlib registry backup`/`restore` — testable with no AWS account, and it's
the artifact store used by the [Docker deployment](deploy-docker.md).

## One API over both

Your code never branches on the backend:

```python
from ingestlib.storage import artifacts

artifacts.list_documents()                                 # the registry, as metadata
artifacts.document_markdown(doc_id)                        # whole-doc markdown (bytes)
artifacts.read_blob(artifacts.page_image_key(doc_id, 1))   # a page PNG — s3 or local
artifacts.delete_document(doc_id)
```

## Why artifacts matter

The artifact store keeps a document's **bytes** — source, page renders, figure
crops, whole-doc markdown — while the **registry** keeps everything queryable and
is the source of truth for structure. Nothing about your corpus is locked inside
the vector database: the chunks live in the registry, the bytes in the artifact
store, so [`reindex()`](manage-corpus.md#rebuild-the-vector-store-reindex)
re-embeds straight from the registry — no re-parse — to rebuild a vector store.

---

Next: [Switch AI providers](switch-providers.md).
