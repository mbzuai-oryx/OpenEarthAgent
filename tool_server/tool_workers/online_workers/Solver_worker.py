import argparse
import uuid
import io
import copy
from typing import Any, Optional
from func_timeout import func_set_timeout
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"solver_worker_{worker_id}.log")


class GenericRuntime:
    def __init__(self, global_dict: Optional[dict] = None, headers: list = []):
        self._global_vars = copy.copy(global_dict) if global_dict else {}
        for c in headers:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars, self._global_vars)


class SolverWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="Solver",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 timeout=20,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,   # no model_path
            None,   # no tokenizer
            model_name,
            False,  # load_8bit
            False,  # load_4bit
            "cpu",  # always run on CPU
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )
        self.timeout = timeout

    def init_model(self):
        logger.info("SolverWorker does not load a model, Ready to run.")

    def generate(self, params):
        if "command" not in params:
            txt_e = "Missing required parameter: command"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}
        
        code = params.get("command")

        try:
            # strip markdown fences
            if code.startswith("```python") and code.endswith("```"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```") and code.endswith("```"):
                code = code.split("```")[1].split("```")[0]
            if code.startswith("python"):
                code = code.split("python")[1]
            # else:
            #     txt_e = "Invalid code format: expected fenced block with ```python or ```"
            #     logger.error(txt_e)
            #     return {"text": txt_e, "error_code": 4}

            txt = func_set_timeout(self.timeout)(self._call)(code)
            if txt is None:
                txt_e = "Execution returned None"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}
            return {"text": txt, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in Solver: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def _call(self, code: str) -> str:
        runtime = GenericRuntime(headers=['from sympy import symbols, Eq, solve'])
        runtime.exec_code(code)
        return str(runtime.eval_code("solution()"))

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "Solver",
                "description": "Execute Python code that defines a `solution()` returning the solution of equations using sympy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Python code in Markdown fenced format defining a `solution()` that uses sympy to solve equations and returns a string."
                        }
                    },
                    "required": ["text"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20009)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="Solver")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SolverWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    worker.run()
