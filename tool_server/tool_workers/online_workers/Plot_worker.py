import argparse
import uuid
import io
import os
import copy
from PIL import Image
from typing import Any, Optional
from func_timeout import func_set_timeout
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"plot_worker_{worker_id}.log")

class GenericRuntime:
    def __init__(self, global_dict: Optional[dict] = None, headers: list = []):
        self._global_vars = copy.copy(global_dict) if global_dict else {}
        for c in headers:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars, self._global_vars)


class PlotWorker(BaseToolWorker):
    def __init__(self,
                controller_addr,
                worker_addr="auto",
                worker_id=worker_id,
                no_register=False,
                model_name="Plot",
                host="0.0.0.0",
                port=None,
                limit_model_concurrency=5,
                model_semaphore=None,
                save_path = None,
                wait_timeout=120.0,
                task_timeout=30.0,
                timeout=20,
                ):
        self.save_path = save_path
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
        logger.info("PlotWorker does not load a model, Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

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
            
            res = func_set_timeout(self.timeout)(self._call)(code)
            if res is None:
                txt_e = "Execution returned None"
                logger.error(txt_e)
                return {"text": "Execution returned None", "error_code": 4}
                
            # save result image
            new_filename = f"plot_{uuid.uuid4().hex[:8]}.png"
            if self.save_path and os.path.isdir(self.save_path):
                save_path = os.path.join(self.save_path, new_filename)
            else:
                if not os.path.isdir(self.save_path):
                    logger.warning(f"Save path '{self.save_path}' is not a valid directory. "
                                f"Falling back to default ./tools_output/")
                save_dir = os.path.join(os.getcwd(), "tools_output")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, new_filename)
            
            res.save(save_path)
            
            txt = f"Plot saved to {new_filename}"
            return {"text": txt, "image": save_path, "error_code": 0}

        except Exception as e:
            txt_e = f"Error in Plot: {e}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 1}

    def _call(self, code: str) -> Image.Image:
        from matplotlib.pyplot import Figure

        runtime = GenericRuntime(headers=['import matplotlib.pyplot as plt'])
        runtime.exec_code(code)
        figure: Figure = runtime.eval_code("solution()")
        if not isinstance(figure, Figure):
            raise TypeError("The `solution` function must return a matplotlib Figure.")

        buf = io.BytesIO()
        figure.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "Plot",
                "description": "Execute Python code that defines a `solution()` returning a matplotlib figure and return the plotted diagram.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Python code in Markdown fenced format defining a `solution()` that returns a matplotlib figure."
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
    parser.add_argument("--port", type=int, default=20005)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="Plot")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = PlotWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        save_path=args.save_path,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    worker.run()
