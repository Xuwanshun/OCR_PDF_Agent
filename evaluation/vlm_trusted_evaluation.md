# OCR System Trusted Evaluation Report

## Scope
- Mode: Full OCR system index (`chroma_db`) with visual extraction enabled
- Dataset: 5 official PDFs in `Document/`
- Focus: extraction quality + retrieval grounding for visual queries

## Extraction Quality
- Docs / pages: 5 / 879
- Region extraction coverage: 191/191 (1.0)
- Table headers populated: 44/96 (0.458)
- Table rows populated: 49/96 (0.51)
- Chart axes populated: 95/95 (1.0)
- Chart series populated: 0/95 (0.0)
- Chart trend populated: 0/95 (0.0)

## Retrieval Quality (12 visual-focused cases)
- Expected type @ top1: 7/12 (0.583)
- Expected type @ top6: 9/12 (0.75)
- Any visual chunk @ top6: 9/12 (0.75)

## Trust Notes
- This report uses deterministic checks (schema-population and retrieval typing) rather than LLM-as-judge scoring.
- Retrieval uses embedding search over your persisted VLM index and reports chunk-type grounding directly.
- If needed, add a human-labeled QA set for factual exactness scoring (EM/F1) as a next step.
