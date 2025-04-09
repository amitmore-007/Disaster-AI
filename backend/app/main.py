
from fastapi import FastAPI, Request
from pydantic import BaseModel
from planner import generate_disaster_response


app = FastAPI()

class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/generate-plan")
async def generate_plan(request: ScenarioRequest):
    plan = generate_disaster_response(request.scenario)
    return {"response_plan": plan}
