from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    text: str
    bbox: list[float] = Field(description="[x0,y0,x1,y1] in PDF coordinate space")
    confidence: float = 1.0
    source: Literal["pdf", "ocr"] = "pdf"


class RegionExtractionTable(BaseModel):
    headers: Optional[list[str]] = None
    rows: Optional[list[list[Any]]] = None
    notes: Optional[str] = None
    confidence: Optional[float] = None
    units: Optional[str] = None


class RegionExtractionChart(BaseModel):
    chart_type: Optional[str] = None
    axes: Optional[dict[str, Any]] = None
    series: Optional[list[dict[str, Any]]] = None
    data_points: Optional[list[dict[str, Any]]] = None
    trend_summary: Optional[str] = None
    notes: Optional[str] = None


class Region(BaseModel):
    region_id: int
    type: Literal["table", "chart", "figure"]
    page: int
    bbox: list[float]
    image_path: Optional[str] = None
    extraction: Optional[dict[str, Any]] = None


class PageContent(BaseModel):
    page_number: int
    width: float
    height: float
    blocks: list[TextBlock] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)


class DocumentJSON(BaseModel):
    version: str = "0.1"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    doc_hash: str
    source_path: str
    pages: list[PageContent]


class IngestMetadata(BaseModel):
    version: str = "0.1"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    doc_hash: str
    source_path: str
    pages: int
    vlm_enabled: bool = True
    ocr_engine: str
