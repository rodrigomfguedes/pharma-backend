from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.tools import StructuredTool
import requests

# API Key
load_dotenv()

# --- Adverse Events Tool

def get_adverse_events(drug_name: str, limit: int = 100) -> str:
    drug_name = drug_name.upper()
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    adverse_effects_counter = {}
    for patient in data.get("results", []):
        for reaction in patient["patient"]["reaction"]:
            name = reaction["reactionmeddrapt"]
            adverse_effects_counter[name] = adverse_effects_counter.get(name, 0) + 1

    top_20 = sorted(adverse_effects_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    formatted = "\n".join(f"{effect}: {count}" for effect, count in top_20)

    return f"Top 10 adverse effects for {drug_name}:\n{formatted}"

class AdverseEventsInput(BaseModel):
    drug_name: str

adverse_events_tool = StructuredTool.from_function(
    func=get_adverse_events,
    name="AdverseEventsRetriever",
    description="Gets the top 10 FDA adverse events for a given drug name.",
    args_schema=AdverseEventsInput,
)