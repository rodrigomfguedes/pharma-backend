# --- Neo4j Tool
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.tools import StructuredTool
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

load_dotenv()





def query_neo4j(question: str) -> str:
    NEO4J_URI="bolt://3.95.30.19"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="departures-octobers-justice"
    NEO4J_DATABASE="neo4j"


    # Connect to Neo4j DB
    graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )

    # Get Neo4j Database Schema
    schema = graph.schema

    # Seu prompt original para geração de Cypher
    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"],
        template="""Task:Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.
        After identifying the drug to which the question is related to, capitalize all letters of it.

        Schema:
        {schema}

        Write multiple Cypher queries , corresponding to all of the possible paths between nodes, if multiple are possible.
        Example - connecting Manufacturer with Drug:
        1) (m:Manufacturer)-[:REGISTERED]->(c:Case)-[:IS_PRIMARY_SUSPECT]->(d:Drug)
        2) (m:Manufacturer)-[:REGISTERED]->(c:Case)-[:IS_CONCOMITANT]->(d:Drug)
        3) (m:Manufacturer)-[:REGISTERED]->(:Case)-[:IS_SECONDARY_SUSPECT]->(:Drug)

        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.

        The question is:
        {question}"""
        )


    # Inicializa o chain com os ajustes:
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-4.1-nano-2025-04-14"),
        qa_llm=ChatOpenAI(temperature=0, model="gpt-4.1-nano-2025-04-14"),
        graph=graph,
        verbose=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        allow_dangerous_requests=True,
        return_direct=True, # Avoids model from replying idk even if it gets right context
        return_intermediate_steps=True,
        top_k=10, # Limits results,
        validate_cypher=True,  # Prevents wrongful cypher syntax/queries
        timeout=10 # Raises connection timeouts
        )
    return chain.invoke({"query": question})


class Neo4jQueryInput(BaseModel):
    question: str

neo4j_tool = StructuredTool.from_function(
    func=query_neo4j,
    name="Neo4jTool",
    description="""Answers questions about the Neo4j Healthcare Analytics template,
                   which contains information about Drugs, Manufacturers, Therapies, Outcomes, and Reactions.
                   After identifying the drug to which the question is related to, capitalize all letters of it. Example: furosemide -> FUROSEMIDE this is Very Important!!! 
                   Example question: Which manufacturers are connected to drugs that contain Furosemide in its name?
                   
                   Output Restrictions:
                   The queries are limited to 10 results, for cost management purposes.""",
    args_schema=Neo4jQueryInput,
)