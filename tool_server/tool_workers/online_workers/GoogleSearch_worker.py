import argparse
import uuid
import os
import requests

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"google_search_worker_{worker_id}.log")


class GoogleSearchWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 api_key="ENV",
                 search_type="search",
                 timeout=5,
                 max_out_len=1500,
                 with_url=False,
                 model_name="GoogleSearch",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None, None, model_name,
            False, False, "cpu",
            limit_model_concurrency,
            host, port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

        if api_key == "ENV":
            api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("Missing Serper API key. Set SERPER_API_KEY in env or pass via --api-key.")

        self.api_key = api_key
        self.search_type = search_type
        self.timeout = timeout
        self.max_out_len = max_out_len
        self.with_url = with_url

    def init_model(self):
        logger.info("GoogleSearchWorker does not load a model. Ready to query Serper API.")

    def generate(self, params):
        if "query" not in params:
            txt_e = "Missing required parameter: query"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        query = params.get("query")
        k = int(params.get("k", 10))

        try:
            logger.info(f"Searching Google for query='{query}', top {k} results")
            status_code, results = self._search(query)
            if status_code != 200:
                txt_e = f"Serper API error {status_code}: {results}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            txt = self._parse_results(results, k)
            return {"text": txt, "error_code": 0}

        except Exception as e:
            txt_e =f"Error in GoogleSearch: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def _search(self, query: str):
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        response = requests.post(
            f"https://google.serper.dev/{self.search_type}",
            headers=headers,
            json={"q": query},
            timeout=self.timeout
        )
        return response.status_code, response.json()

    def _parse_results(self, results: dict, k: int) -> str:
        snippets = []

        answer_box = results.get("answerBox", {})
        if answer_box:
            if answer_box.get("answer"):
                snippets.append(f"Answer box: {answer_box['answer']}")
            elif answer_box.get("snippet"):
                snippets.append(f"Answer box: {answer_box['snippet']}")

        kg = results.get("knowledgeGraph", {})
        if kg:
            desc = f"{kg.get('title','')} knowledge graph: {kg.get('type','')}. {kg.get('description','')}"
            if kg.get("attributes"):
                attrs = ', '.join(f"{k}: {v}" for k,v in kg["attributes"].items())
                desc += f" ({attrs})"
            snippets.append(desc)

        for item in results.get("organic", [])[:k]:
            content = ""
            if item.get("title"): content += item["title"] + ": "
            # if item.get("link") and self.with_url: content += f"({item['link']}) "
            if item.get("snippet"): content += item["snippet"]
            snippets.append(content)

        if not snippets:
            return "No good Google Search result found."

        result = ""
        for idx, item in enumerate(snippets):
            result += f"{idx+1} - {item.strip().replace(chr(10), ' ')}\n\n"

        return result[:self.max_out_len]

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "GoogleSearch",
                "description": "Searches the input query from Google and returns the top results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "k": {"type": "integer", "description": "Number of top results to return (default 10)."}
                    },
                    "required": ["query"],
                    "optional": ["k"]
                }
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20006)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--api-key", type=str, default="ENV")
    parser.add_argument("--search-type", type=str, default="search")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = GoogleSearchWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        api_key=args.api_key,
        search_type=args.search_type,
        host=args.host,
        port=args.port,
        limit_model_concurrency=args.limit_model_concurrency,
    )
    worker.run()
