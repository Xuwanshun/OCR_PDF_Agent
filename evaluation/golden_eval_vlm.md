# Golden QA Evaluation (Full OCR System)

## Method
- Fixed golden question set with expected docs/types and required terms.
- Deterministic scoring on retrieval hit, term recall, citation validity, and abstention.
- Optional LLM-as-judge score is reported as secondary evidence.

## Summary
- Questions: 10
- Deterministic pass: 9/10 (0.9)
- Expected doc hit: 10/10 (1.0)
- Expected type hit: 10/10 (1.0)
- Mean term recall: 0.85
- Citation valid rate (mean): 0.7
- Abstain count: 1/10

## Per Question
- Q1 (text): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=1.0 abstain=False
- Q2 (text): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=1.0 abstain=False
- Q3 (text): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=0.5 abstain=False
- Q4 (text): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=1.0 abstain=False
- Q5 (visual_table): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=0.5 abstain=False
- Q6 (visual_table): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=0.5 abstain=False
- Q7 (visual_chart): pass=False doc_hit=True type_hit=True term_recall=0.0 citation_valid=0.0 abstain=True
- Q8 (visual_chart): pass=True doc_hit=True type_hit=True term_recall=0.5 citation_valid=1.0 abstain=False
- Q9 (citation): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=0.5 abstain=False
- Q10 (citation): pass=True doc_hit=True type_hit=True term_recall=1.0 citation_valid=1.0 abstain=False
