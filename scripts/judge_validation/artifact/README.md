# Judge-validation labeling artifact

Hand labels for validating the LLM judges, collected on 2026-11-06 with a
claude.ai artifact page (https://claude.ai/artifact/52Zm1tJLJcrK5gbqLZYVzj).

- `judge-labeling.html`: the page, as published (version 1).
- `items.json`, `judge.json`: the data files served with the page, written by
  `build_artifact_data.py` from the manifests drawn by `sample_items.py`.
  `items.json` is exactly what annotators saw; `judge.json` holds the judge's
  answers, shown only on the progress page.
- `labels.jsonl`: every saved label, one line per (annotator, item), exported
  from the page's database after labeling finished. Each line carries the
  rubric hash, schema version and app version it was saved under, plus any
  earlier answers for that item in `history`.
- `practice_labels.jsonl`: practice-mode labels, which do not count.

Both annotators labeled all 120 sampled items (8 categories x 15). The `queue`
field records which queue an item was labeled from (a category queue or one of
the mixed queues); it does not affect the label.
