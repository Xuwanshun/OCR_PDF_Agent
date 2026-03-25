from ocr_agent.costs import CostEstimate, DirectoryStats, estimate_costs


def test_estimate_costs_basic():
    stats = DirectoryStats(
        pdf_count=1,
        total_pages=10,
        total_chars=40000,
        text_chunk_count=8,
        table_like_region_count=2,
        figure_region_count=3,
    )
    estimate = estimate_costs(
        stats=stats,
        embed_model="text-embedding-3-small",
        vision_model="gpt-4o-mini",
        qa_model="gpt-4o-mini",
        vlm_input_tokens_per_region=1000,
        vlm_output_tokens_per_region=200,
        qa_input_tokens=2000,
        qa_output_tokens=300,
    )
    assert isinstance(estimate, CostEstimate)
    assert estimate.embedding_cost_usd > 0
    assert estimate.vlm_cost_usd > 0
    assert estimate.total_ingest_cost_usd >= estimate.embedding_cost_usd
    assert estimate.qa_per_question_cost_usd > 0
