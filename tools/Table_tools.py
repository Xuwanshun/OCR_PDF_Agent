from langchain.tols import tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

vlm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
TABLE_ANALYSIS_PROMPT = """You are a Table Analysis specialist.
Analyze this chart/figure image and extract:

"""
def call_vlm_with_image(image_base64: str, prompt: str) -> str:
    """Call VLM with an image and prompt"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    )
@tool
def AnalyzeCaht(region_id: int) -> str:
    """Analyze/Extract a Table region using VLM.
    Use this tool when you need to extract data from tabular data

    Args:
        region_id: The id of the layout region to analyze(must be a table type)
    
    Return:
        JASN string with table headers, rows, and any notes
    """
    if region_id not in region_images:
        return f"Error: Region {region_id} not found"

    region_data = region_image[region_id]

    if region_data['type'] not in ['chat', 'figure']:
        return f"Warning: Region {region_id}"
    
    result = call_vlm_with_image(region_data['base64'], TABLE_ANALYSIS_PROMPT)
    return result
