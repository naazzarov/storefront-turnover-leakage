"""Model adapters for the Yelp text-to-SQL benchmark.

Each adapter exposes a single method ``generate_sql(question) -> str``.
The harness is model-agnostic; add a new model by subclassing ``BaseModel``
and registering it in ``MODELS``.

Dependencies are avoided: LLM calls go through ``urllib`` directly against
an OpenAI-compatible ``/v1/chat/completions`` endpoint, so this works for
both the OpenAI API and a local Ollama server (which exposes the same API).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

SCHEMA = """Table: business_small
Columns:
- business_id (VARCHAR) - Unique identifier
- name (VARCHAR) - Business name
- city (VARCHAR) - City name
- state (VARCHAR) - State abbreviation
- stars (FLOAT) - Rating 0-5
- review_count (INTEGER) - Number of reviews
- categories (VARCHAR) - Business categories (comma-separated)

Sample cities: Philadelphia, Phoenix, Tampa, Toronto, Tucson, Pittsburgh, Orlando, Mesa, Ottawa, Las Vegas
Sample states: PA, AZ, FL, ON, NV
Sample categories: Restaurants, Bars, Cafes, Shopping, Automotive, Beauty & Spas, Health & Medical, Home Services, Nightlife, Fast Food
"""

SYSTEM_PROMPT = f"""You are a PostgreSQL expert. Generate a valid SQL query based on the natural language question.

RULES (MANDATORY):
1. ONLY return the SQL query
2. NO explanations, NO markdown, NO code blocks
3. ONLY SELECT queries allowed
4. Do NOT invent columns that don't exist
5. Use exact table name: business_small
6. Add LIMIT 100 if not specified
7. Use ORDER BY for rankings (descending)
8. Use GROUP BY for aggregations
9. Ensure valid SQL syntax

DATABASE SCHEMA:
{SCHEMA}

OUTPUT: Only the SQL query, nothing else."""


def clean_sql(sql: str) -> str:
    sql = re.sub(r"```(?:sql)?", "", sql)
    sql = sql.strip()
    sql = sql.rstrip(";")
    return sql


class BaseModel:
    name = "base"

    def generate_sql(self, question: str) -> str:
        raise NotImplementedError


class RuleBasedBaseline(BaseModel):
    """A deterministic, non-LLM baseline.

    Maps a handful of common question patterns to SQL templates. It is
    intentionally simple: it serves as a lower-bound reference point so the
    value added by LLMs is measurable rather than assumed.
    """

    name = "rule-based"

    def generate_sql(self, question: str) -> str:
        q = question.lower()
        if "how many" in q or "count" in q:
            return "SELECT COUNT(*) FROM business_small"
        if "average" in q or "avg" in q:
            return "SELECT AVG(stars) FROM business_small"
        if "top" in q or "highest" in q or "most" in q:
            return "SELECT name, stars FROM business_small ORDER BY stars DESC LIMIT 5"
        if "lowest" in q or "bottom" in q:
            return "SELECT name, stars FROM business_small ORDER BY stars ASC LIMIT 5"
        return "SELECT name FROM business_small"


class _OpenAICompatibleClient(BaseModel):
    def __init__(self, base_url: str, api_key: str | None, model: str, label: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = label

    def _call(self, question: str) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def generate_sql(self, question: str) -> str:
        raw = self._call(question)
        return clean_sql(raw)


class OpenAIModel(_OpenAICompatibleClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        super().__init__("https://api.openai.com/v1", key, model, f"openai/{model}")


class OllamaModel(_OpenAICompatibleClient):
    def __init__(self, model: str = "llama3.1", base_url: str | None = None):
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        super().__init__(url, api_key="ollama", model=model, label=f"ollama/{model}")


class MockModel(BaseModel):
    """Returns canned SQL for offline harness testing. Never used in results."""

    name = "mock"

    def __init__(self, responses: dict[int, str] | None = None):
        self.responses = responses or {}

    def generate_sql(self, question: str) -> str:
        return self.responses.get(0, "SELECT name FROM business_small")


def build_models() -> list[BaseModel]:
    models: list[BaseModel] = [RuleBasedBaseline()]
    try:
        models.append(OpenAIModel())
    except ValueError:
        pass
    models.append(OllamaModel())
    return models
