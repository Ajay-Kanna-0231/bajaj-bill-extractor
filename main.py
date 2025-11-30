import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64
import json
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY not found in .env file")

# --- Configure Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-1.5-flash"

app = FastAPI()

# ---------- Helper: Convert PIL → base64 ----------
def pil_to_b64(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- Helper: LLM extraction ----------
def extract_with_llm(img: Image.Image):
    img_b64 = pil_to_b64(img)

    prompt = """
Extract ONLY line items from this invoice.

Rules:
1. Ignore totals, subtotals, taxes, final amounts.
2. Extract ONLY true line items: item_name, item_rate, item_quantity, item_amount.
3. item_name must be EXACT as printed.
4. Return ONLY JSON:

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

    result = genai.GenerativeModel(MODEL).generate_content(
        [
            prompt,
            {"mime_type": "image/png", "data": base64.b64decode(img_b64)}
        ],
        generation_config={"response_mime_type": "application/json"}
    )

    data = json.loads(result.text)

    usage = {
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens
    }

    return data, usage


# ---------- API ENDPOINT ----------
@app.post("/extract-bill-data")
async def extract(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw)).convert("RGB")

        data, usage = extract_with_llm(img)

        resp = {
            "is_success": True,
            "token_usage": {
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"]
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

        return resp

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"is_success": False, "message": str(e)}
        )
