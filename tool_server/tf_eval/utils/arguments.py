from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence
from torch.utils.data import Dataset
import transformers
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments
from box import Box

from .utils import *

@dataclass
class ModelArguments:
    model: Optional[str] = field(default="qwen3")
    model_args: Optional[str] = field(default="pretrained=Qwen/Qwen3-4B-Instruct-2507")
    model_mode: Optional[str] = field(default="general")
    batch_size: Optional[int] = field(default=1)
    stop_token: Optional[str] = field(default=None)
    max_rounds: Optional[int] = field(default=3)

@dataclass
class TaskArguments:
    task_name: Optional[str] = field(default="rsagent")
    resume_from_ckpt: Optional[str] = field(default=None,)
    save_to_ckpt: Optional[str] = field(default=None,)
    task_type: Optional[str] = field(default="e2e")
    cache_dir: Optional[str] = field(default=None,)
    save_dir: Optional[str] = field(default=None,)

    def __post_init__(self):
        for attr in ["resume_from_ckpt", "save_to_ckpt"]:
            value = getattr(self, attr)
            if value is None:
                setattr(self, attr, Box())
            elif isinstance(value, str) and "=" in value:
                key, path = value.split("=", 1)
                setattr(self, attr, Box({key.strip(): path.strip()}))
            else:
                raise ValueError(
                    f"{attr} must be a string in 'key=/path/to/file' format or None"
                )


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the Evaluation script.
    """
    config: Optional[str] = field(default=None)
    verbosity: Optional[str] = field(default="INFO")
    wandb_args: Optional[str] = field(default="project=mr_eval,entity=mr_eval")
    output_path: Optional[str] = field(default="output")
    controller_addr: Optional[str] = field(default=None)


def parse_str_into_dict(args_str: str) -> Dict:
    """
    Parse a string of comma-separated key-value pairs into a dictionary.
    """
    args_dict = {}
    for arg in args_str.split(","):
        key, value = arg.split("=")
        args_dict[key] = value
    return args_dict

def parse_str_into_list(args_str: str) -> List:
    """
    Parse a string of comma-separated values into a list.
    """
    # import pdb; pdb.set_trace()
    return args_str.split(",")

def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TaskArguments, ScriptArguments))
    # breakpoint()
    model_args, task_args, script_args = parser.parse_args_into_dataclasses()
    
    if script_args.config:
        if script_args.config.endswith(".json"):
            config = load_json_file(script_args.config)
        elif script_args.config.endswith(".yaml"):
            config = load_yaml_file(script_args.config)
        else:
            raise ValueError("Config file should be either a json or yaml file.")
        
        if isinstance(config, dict):
            model_args = ModelArguments(**config["model_args"])
            task_args = TaskArguments(**config["task_args"])
            script_args = ScriptArguments(**config["script_args"])
        elif isinstance(config, list):
            model_args = ModelArguments(**config[0]["model_args"])
            task_args = TaskArguments(**config[0]["task_args"])
            script_args = ScriptArguments(**config[0]["script_args"])
        else:
            raise ValueError("Config file should be either a dict or list of dicts.")
    else:
        config = None
        
    # import pdb; pdb.set_trace() 
    script_args.config = config
    task_args.task_name = parse_str_into_list(task_args.task_name)
    if isinstance(model_args.model_args, str):
        model_args.model_args = parse_str_into_dict(model_args.model_args)
    if isinstance(script_args.wandb_args, str):
        script_args.wandb_args = parse_str_into_dict(script_args.wandb_args)
    
    return dict(model_args=model_args, task_args=task_args, script_args=script_args)
    
