# This file patches in a multipart endpoint without refactoring the whole app.
# Import in main.py at bottom.

from fastapi import UploadFile, File, Form, HTTPException
from pydantic import TypeAdapter
import json

from app.main import api, plan as plan_json_endpoint
from app.models import PlanRequest

@api.post("/plan_multipart")
async def plan_multipart(
    payload: str = Form(...),
    screenshot: UploadFile = File(...)
):
    try:
        obj = json.loads(payload)
        req = TypeAdapter(PlanRequest).validate_python(obj)
        return await plan_json_endpoint(req, screenshot)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
