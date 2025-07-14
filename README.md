# Pharma Chatbot

This project is an example of a Chatbot that uses an agent with access to 3 tools:

- Neo4j database query Agent
- Grunenthal's 24/25 financial year review (P&L)
- FDA's adverse Events

This project uses the folowing tech stack:

- [FastAPI](https://fastapi.tiangolo.com/) - High performance, easy to learn, fast to code, ready for production framework.
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) - A high-performance Python library for data extraction, analysis, conversion & manipulation of PDF (and other) documents.
- [LangChain](https://www.langchain.com/) - A full product suite for reliable agents and LLM apps · Trace and evaluate any LLM app · Build agents any way you want, then deploy and scale with ease.
- [Neo4j](https://www.neo4j.com) - Graph Database & Analytics — Built by data scientists for data scientists. Improve models and sharpen predictions.

## Installation

You will need [PyPI](https://pypi.org/project/pip/) at the latest version configured in your machine before proceding.

Clone the project, go to the root folder and run the following command:

`pip install -r requirements.txt`

## Running the project in the local environment

### Running the application

To run the project in your local environment, after installing the dependencies. In the root folder, run the command:

`uvicorn main:app`

The application should start and you should be able to access it at [http://localhost:8000](http://localhost:8000) in your browser.

Access the /docs endpoint to retrieve all the available endpoints
