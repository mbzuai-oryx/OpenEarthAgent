import itertools
import json
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch, gc
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist

from .models import get_model
from .tasks import get_task_object, get_task_functions
from .tasks.base_dataset.base_evaluation_dataset import BaseEvalDataset, DataCollatorForSupervisedDataset

from .utils.utils import *
from .utils.arguments import *

from .utils.log_utils import get_logger, set_verbosity
# from .utils.evaluate import evaluate_metric
from .tool_inferencer import BaseToolInferencer
import pdb
import re
import os 

try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify package not found. Please install it to use math verification features.")

logger = get_logger(__name__)

class TFEvaluator():
    def __init__(self, model_args, task_args, script_args):
        print("initializing evaluator")
        self.config = script_args.config
        self.model_args = model_args
        self.task_args = task_args
        self.script_args = script_args
        self.tasks = self.task_args.task_name
        self.model = get_model(self.model_args.model)(**self.model_args.model_args)
        max_rounds = self.model_args.max_rounds
        stop_token = self.model_args.stop_token
        print(self.model,self.tasks,self.script_args)
        set_verbosity(self.script_args.verbosity)
        
        self.inferencer = BaseToolInferencer(
            tp_model=self.model,
            batch_size=self.model_args.batch_size,
            model_mode=self.model_args.model_mode,
            max_rounds = max_rounds,
            stop_token = stop_token,
            controller_addr = self.script_args.controller_addr,
            cache_dir=self.task_args.cache_dir,
            save_dir=self.task_args.save_dir,
        )
        print("initializing evaluator done")

    def log_inference_time(self, task_name, task_config, start_time, ckpt_path, ckpt_samples, interrupted=False):
        """Compute elapsed time, per-sample rate, and save to timing log file."""
        elapsed = time.time() - start_time
        if ckpt_path and os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                new_sample_count = sum(1 for _ in f)
        else:
            new_sample_count = 0
        num_samples_processed= new_sample_count - ckpt_samples
        per_sample = (elapsed / num_samples_processed if num_samples_processed > 0 else float("nan"))

        prefix = "[INTERRUPTED]" if interrupted else "[COMPLETED]"
        time_info = (
            f"{prefix} Task={task_name} | "
            f"Type={task_config.task_type} | "
            f"Model={self.model_args.model} | "
            f"ModelArgs={self.model_args.model_args} |"
            f"Old_samples_len={ckpt_samples} | "
            f"new_samples_len={new_sample_count} | "
            f"ProcessedSamples={num_samples_processed} | "
            f"InferenceTime={elapsed:.2f}s | "
            f"InferenceTimePerSample={per_sample:.2f}s\n"
        )

        logger.info(time_info)
        timing_log_path = "./tool_server/tf_eval/scripts/logs/inference_time_log.txt"
        os.makedirs(os.path.dirname(timing_log_path), exist_ok=True)
        with open(timing_log_path, "a", encoding="utf-8") as f:
            f.write(time_info)

        return elapsed, per_sample
        
    def evaluate(self):

        for task_name in self.tasks:
            logger.info(f"evaluating {task_name}")
            task_dict = get_task_functions(task_name)
            load_data_function, evaluate_function, task_config = task_dict["load_data_function"], task_dict["evaluate_function"], task_dict["task_config"]
            self.model.set_generation_config(task_config.generation_config)

            # overwrite task_config.task_type from task_args
            if self.task_args.task_type in ["e2e", "step"]:
                 task_config.task_type = self.task_args.task_type

            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_fn,
                evaluate_function=evaluate_function,
                task_config = task_config,
                task_args = self.task_args,
                model_args = self.model_args,
            )

            start_time = time.time()
            ckpt_path = self.task_args.resume_from_ckpt.get(task_name, None)

            if ckpt_path and os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    ckpt_samples = sum(1 for _ in f)
            else:
                ckpt_samples = 0

            try:
                if task_config.task_type == "e2e":
                    self.inferencer.batch_inference(dataset)     # dynamic batch
                else:
                    self.inferencer.step_inference(dataset)      # single turn  
            except KeyboardInterrupt:
                logger.warning("⚠️ Ctrl+C detected — saving inference time info before exiting...")
                self.log_inference_time(task_name, task_config, start_time, ckpt_path, ckpt_samples, interrupted=True)
                logger.info("Inference time saved. Exiting\n")
                return 
            
            # --- If finished normally ---
            self.log_inference_time(task_name, task_config, start_time, ckpt_path, ckpt_samples, interrupted=False)

            # breakpoint()
            res_log = dataset.evaluate()
            if is_main_process() or "vllm_models" in self.model_args.model:
                logger.info(f"evaluation of {task_name} completed")
                json_file_path = f"{self.script_args.output_path}/{ckpt_path.split('/')[-1].split('.')[0]}.json"
                write_json_file(res_log,json_file_path )
            
