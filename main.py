import os
import json
import base64
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY not found.")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "models/gemini-2.5-flash"

app = FastAPI()

def pil_to_b64(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def extract_with_llm(img: Image.Image):
    img_b64 = pil_to_b64(img)

    prompt = """
Extract ONLY line items from this invoice.

Rules:
1. Ignore totals, subtotals, taxes, discounts, and grand total rows.
2. Extract ONLY true bill line items.
3. item_name must be EXACT as printed.
4. Return ONLY valid JSON in this structure:

{
 "page_type": "Bill Detail",
 "bill_items": [
   {
     "item_name": "",
     "item_rate": 0.0,
     "item_quantity": 0.0,
     "item_amount": 0.0
   }
 ]
}
"""

    model = genai.GenerativeModel(MODEL)

    result = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/png",
                "data": base64.b64decode(img_b64)
            }
        ],
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json"
        }
    )

    return json.loads(result.text)

@app.post("/extract-bill-data")
async def extract_bill_data(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        img = Image.open(BytesIO(file_bytes)).convert("RGB")

        data = extract_with_llm(img)

        return {
            "is_success": True,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "data": {
                "pagewise_line_items": [
                    {
                        "page_no": "1",
                        "page_type": data.get("page_type", "Bill Detail"),
                        "bill_items": data.get("bill_items", [])
                    }
                ],
                "total_item_count": len(data.get("bill_items", []))
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "is_success": False,
                "message": f"Error: {str(e)}"
            }
        )
