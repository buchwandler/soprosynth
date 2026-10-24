---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.3
kind: fixed
summary:
  Fixed text segmentation to hard-split overlong words at the configured character
  bound
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - soprosynth/api.py
  - soprosynth/postprocess.py
  - tests/test_postprocess.py
  - tests/test_sopro_orchestration.py
  - tests/test_streaming.py
issues: []
prs: []
sources:
  - git:477fd08cdf02d183cbea92d94fe2773bba8eaaed
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
