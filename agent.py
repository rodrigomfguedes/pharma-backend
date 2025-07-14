from agent_tools.adverse_effects import adverse_events_tool
from agent_tools.neo4j import neo4j_tool
from agent_tools.financial import financial_report_tool 
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI  
from dotenv import load_dotenv
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableLambda


logger = logging.getLogger(__name__)

# Get OpenAI key
load_dotenv()

# ----- LLM and Agent

role_description = """
You are an experienced pharmaceutical data agent with access to three specialized tools:
1. **Neo4jTool** – for querying drug-related data (manufacturers, therapies, outcomes, reactions) from a Neo4j healthcare analytics graph.
2. **AdverseEventsRetriever** – for retrieving the top 10 FDA adverse events for a given drug via FDA API.
3. **FinancialReportRetriever** – for retrieving Grunenthal's financial report data 24/25 and overall company performance.

Always pass the drug name in UPPERCASE. Example: Tylenol -> TYLENOL. This is VERY IMPORTANT!!

When a user asks about drug relationships (e.g. manufacturers, therapies, outcomes), use **Neo4jTool**. Example question: Which manufacturers are connected to drugs which contain Tylenol in its name?
When a user asks about adverse effects, use **AdverseEventsRetriever**, use **AdverseEventsRetriever**. Example question: What are the top 10 most recent adverse events registered for a drug containing Tramadol in its name?
When a user asks about Grunenthal, use **FinancialReportRetriever**.  

Be very direct and precise in answering the question, reply only to what the user asked, don't add additional information. THIS IS VERY IMPORTANT!!! 

If the question is beyond your tools’ scope, reply exactly: 
"I can't help you with that. I'm equipped with relevant pharmaceutical data to provide you with evidence‑based insights in this field."
"""


agent = initialize_agent(
    tools=[neo4j_tool, adverse_events_tool, financial_report_tool],
    llm=OpenAI(temperature=0.02, model="gpt-4.1-nano-2025-04-14"),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,  # importante,
    timeout=30,
    agent_kwargs={
        "prefix": role_description,
    },
)


MAX_RETRIES = 10
TIMEOUT_SECONDS = 60


def safe_run_agent(query: str, max_retries: int = MAX_RETRIES, timeout: int = TIMEOUT_SECONDS):
    """
    Executes the agent with a retry mechanism and timeout protection.

    Parameters:
        query (str): The input question or command to send to the agent.
        max_retries (int): The maximum number of attempts to retry the agent execution.
        timeout (int): Timeout duration in seconds for each attempt.

    Returns:
        dict: The result returned by the agent, including 'output' and optionally 'intermediate_steps'.

    Raises:
        RuntimeError: If all attempts fail, including due to timeout or unexpected exceptions.
    """
    for attempt in range(1, max_retries + 1):
        print(f"\nAttempt {attempt} of {max_retries}...")

        try:
            # Execute the agent in a separate thread to apply a timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(agent, {"input": query})
                result = future.result(timeout=timeout)

            # Check if the agent returned a valid response
            if not result.get("output"):
                print("Invalid or empty response. Retrying...")
                continue

            print("Execution successful.")
            return result

        except FutureTimeoutError:
            print(f"Execution timed out after {timeout} seconds. Retrying...")
        except Exception as e:
            print("Error occurred during agent execution:")
            traceback.print_exc()

    # All retries failed
    raise RuntimeError("All attempts to execute the agent have failed.")
