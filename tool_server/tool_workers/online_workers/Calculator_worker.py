import uuid
import argparse
import addict
import math
from func_timeout import func_timeout
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"Calculator_Worker_{worker_id}.log")


# ------------------------
# Safe eval for calculator
# ------------------------
def safe_eval(expr):
    math_methods = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_methods = {
        "math": addict.Addict(math_methods),
        "max": max,
        "min": min,
        "round": round,
        "sum": sum,
        **math_methods,
    }
    allowed_methods["__builtins__"] = None
    return eval(expr, allowed_methods, allowed_methods)


class CalculatorToolWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="Calculator",
                 model_path="",
                 model_base="",
                 load_8bit=False,
                 load_4bit=False,
                 device="",
                 limit_model_concurrency=1,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 timeout=2):
        self.timeout = timeout
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore
        )

    def init_model(self):
        logger.info(f"Calculator tool initialized with timeout={self.timeout}s")

    def generate(self, params):
        if "expression" not in params:
            txt_e = "Missing required parameter: 'expression' for calculator."
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        expression = params.get("expression")

        try:
            result = func_timeout(self.timeout, safe_eval, [expression])
            return {"text": f"{result}", "error_code": 0}
        except Exception as e:
            txt_e = f"Error in calculator: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20010)  # different default than OCR
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://0.0.0.0:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = CalculatorToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        no_register=args.no_register
    )
    worker.run()
