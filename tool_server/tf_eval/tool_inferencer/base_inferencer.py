
import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
import requests
import re
import copy

from ..models.abstract_model import tp_model
from .dynamic_batch_manager import DynamicBatchManager
from ..utils.utils import *
from ..utils.log_utils import get_logger
from ...tool_workers.tool_manager.base_manager import ToolManager
from ...tool_workers.cache_manager.base_cache_manager import CacheManager
MAX_TOOL_TEXT_CHARS = 3000

import torch.distributed as dist

logger = get_logger(__name__)

class BaseToolInferencer(object):
    def __init__(
        self,
        tp_model: tp_model = None,
        # dataset: Dataset = None,
        batch_size: int = 1,
        model_mode: str = "general",
        max_rounds: int = 3,
        stop_token: str = "<stop>",
        controller_addr: str = "http://0.0.0.0:20001",
        cache_dir: str | None = None,
        save_dir: str | None = None,
    ):
        self.accelerator = Accelerator()
        self.tp_model = tp_model
        self.model_mode = model_mode 
        self.generate_conversation_fn = self.tp_model.generate_conversation_fn
        self.append_conversation_fn = self.tp_model.append_conversation_fn

        if dist.is_initialized() and self.accelerator.device.type == "cuda" and not 'vllm_models' in str(type(self.tp_model)):
            self.tp_model = self.tp_model.to(self.accelerator.device)
            self.tp_model = self.tp_model.to(torch.bfloat16)

        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.stop_token = stop_token
        self.controller_addr = controller_addr
        self.manager = DynamicBatchManager(
            batch_size=self.batch_size, 
            max_rounds=self.max_rounds, 
            stop_token=self.stop_token,
            generate_conversation_fn = self.tp_model.generate_conversation_fn,
        )
        self.tool_manager = ToolManager()
        self.cache_manager = CacheManager(cache_dir,save_dir)
        self.available_models = self.tool_manager.available_tools
        logger.info(f"{len(self.available_models)} available models: {self.available_models}")

    def batch_tool_response_to_next_round_input(self):
        current_batch = self.manager.get_current_batch()
        
        for idx,item in enumerate(current_batch):
            if item.model_response is None or item.status != "processing":
                continue
            # breakpoint() 
            tool_cfg = item.tool_cfg[item.current_round-1]
            tool_response = item.tool_response[item.current_round-1]
            action_parse_error = item.action_parse_error[item.current_round-1]

            assert len(item.tool_cfg) == item.current_round 
            assert len(item.tool_response) == item.current_round 
            assert len(item.action_parse_error) == item.current_round 

            original_prompt = item.meta_data.get("text", "")
            
            if tool_response is not None:
                try:
                    if "gpkg" in tool_response:
                        item.gpkg = tool_response.pop("gpkg")
                    if "image" in tool_response:
                        item.img = tool_response.pop("image")
                   
                    if "text" in tool_response:
                        tool_response_text = tool_response["text"]
                        ###Without feed back
                        # if tool_response["error_code"] == 0:
                        #     tool_response_text = tool_response["text"]
                        # else:
                        #     tool_response_text = "Could not be completed"
                    else:
                        tool_response_text = None
                    # breakpoint() 
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))
                    new_response = f"OBSERVATION:\n{api_name} model outputs: {tool_response_text}\n"
                    new_round_prompt = f"{new_response}Please summarize the model outputs and answer my first question."                        
                except Exception as e:
                    logger.error(
                        f"[batch_tool_response_to_next_round_input] "
                        f"Failed to build new_round_prompt for tool"
                        f"Reason: {e}. Tool response: {str(tool_response)}"
                    )
                    new_round_prompt = original_prompt
            else:
                if action_parse_error:
                    new_round_prompt = action_parse_error + " " + original_prompt
                else:
                    new_round_prompt = original_prompt


            new_round_input = dict(text=new_round_prompt)
            item.new_round_input.append(new_round_input)
            item.conversation = self.append_conversation_fn(
                conversation=item.conversation, text=new_round_prompt, role="user"
            )
            # breakpoint()

    
    def batch_get_tool_response(self):
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            if item.model_response is None or item.status != "processing":
                continue
            
            tool_cfg = item.tool_cfg[item.current_round-1]
            assert len(item.tool_cfg) == item.current_round
            images = item.meta_data.get("images", None)
            if tool_cfg is not None and len(tool_cfg) > 0:
                assert item.status == "processing"
                try:
                    assert len(tool_cfg) == 1, "Only one tool is supported for now, but got: {}".format(tool_cfg)
                    api_name = tool_cfg[0].get("API_name", tool_cfg[0].get("api_name", ""))

                    if api_name not in self.available_models:
                        if api_name == "Terminate":
                            logger.info(f"API_name is {api_name}. Finish!")
                            continue
                        else:
                            logger.error(f"API_name {api_name} not in available models, {self.available_models}")
                            item.tool_response.append(dict(text=f"There is no tool named {api_name}.",error_code=1))
                            continue

                    # if input contains image
                    IMAGE_REQUIRED_TOOLS = [
                        "OCR",
                        "DrawBox",
                        "AddText",
                        "TextToBbox",
                        "CountGivenObject",
                        "ImageDescription",
                        "RegionAttributeDescription",
                        "ChangeDetection",
                        "SegmentObjectPixels",
                        "ObjectDetection",
                        "GetBboxFromGeotiff",
                        "DisplayOnGeotiff"
                    ]
                    GPKG_REQUIRED_TOOLS = [
                        "AddPoisLayer",
                        "ComputeDistance",
                        "DisplayOnMap",
                        "AddIndexLayer",
                        "ComputeIndexChange",
                        "ShowIndexLayer",
                        "DisplayOnGeotiff"
                    ]
                    USE_CACHE_TOOLS = [
                        "GetAreaBoundary",
                        "AddPoisLayer",
                        "ComputeDistance",
                        "AddIndexLayer",
                        "ComputeIndexChange",
                    ]
                    img_1 = pre_image = post_image = gpkg = None

                    if api_name in IMAGE_REQUIRED_TOOLS:
               
                        if api_name == "ChangeDetection":
                            if images is None or len(images) == 0:
                                item.tool_response.append(dict(text=f"Valid image is required for tool",error_code=1))
                                continue
                            assert isinstance(images, list), f"Images must be a list, got {type(images)}"
                            assert all(isinstance(img, str) for img in images), "All elements in 'images' must be strings"
                            assert len(images) == 2, "ChangeDetection requires at least two images: [pre_image, post_image]"
                            pre_image, post_image = images[:2]

                        if not item.img:
                            if images is None or len(images) == 0:
                                item.tool_response.append(dict(text=f"Valid image is required for tool",error_code=1))
                                continue
                            assert isinstance(images, list), f"Images must be a list, got {type(images)}"
                            assert all(isinstance(img, str) for img in images), "All elements in 'images' must be strings"
                            img_1 = images[0]
                        else:
                            img_1 = item.img
     
                    if api_name in GPKG_REQUIRED_TOOLS:
                        assert item.gpkg is not None, "GeoPackage is required for tool: {}".format(api_name)
                        gpkg = item.gpkg

                    api_params = tool_cfg[0].get("api_params", tool_cfg[0].get("API_params", {})).copy()
                    for key in ("image", "pre_image", "post_image", "gpkg"):
                        api_params.pop(key, None)

                    if api_name == "ChangeDetection":
                        api_params["pre_image"] = pre_image
                        api_params["post_image"] = post_image
                    if api_name == "DisplayOnGeotiff" or api_name == "GetBboxFromGeotiff":
                        api_params["geotiff"] = img_1
                    if api_name in IMAGE_REQUIRED_TOOLS:
                        api_params["image"] = img_1
                    if api_name in GPKG_REQUIRED_TOOLS:
                        api_params["gpkg"] = gpkg

                    use_cache_successful = False
                    if api_name in USE_CACHE_TOOLS:
                        tool_response = self.cache_manager.get_response(item,api_name,api_params)
                        if tool_response['error_code'] == 0:
                            use_cache_successful = True
                            logger.info("Use cache successful")
                    if api_name not in USE_CACHE_TOOLS or not use_cache_successful:
                        tool_response = self.tool_manager.call_tool(api_name, api_params)
                    res_text = tool_response['text']    
                    if isinstance(res_text, str) and len(res_text) > MAX_TOOL_TEXT_CHARS:
                        tool_response['text'] =(
                            res_text[:MAX_TOOL_TEXT_CHARS] + " ... "
                        )  
                    tool_response_clone = copy.deepcopy(tool_response)

                    if  tool_response['error_code'] == 0:
                        logger.info(f"The {api_name} calls successfully!")
                    else:
                        logger.info(f"The {api_name} calls failed!")
                    item.tool_response.append(tool_response_clone)
                    continue
                except Exception as e:
                    # import traceback
                    # err_trace = traceback.format_exc()
                    logger.error(f"[Tool Error] {api_name} failed. Reason: {e}\n") 
                    item.tool_response.append(dict(text=f"Tool {api_name} failed : {e}", error_code=1))
                    continue
            else:
                item.tool_response.append(None)
                continue
            
    def extract_actions(self, text: str):
        try:
            actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
            actions_match = re.search(actions_pattern, text)
            if not actions_match:
                return None, "No action found."
            actions_str = actions_match.group(1)
            actions_list = json.loads(actions_str)
            return actions_list, None
            
        except Exception as e:
            msg = f"Invalid action format."
            logger.info(f"{msg}: {e}")
            return None, msg
       
    def batch_parse_tool_config(self):
        current_batch = self.manager.get_current_batch()
        for item in current_batch:
            model_response = item.model_response[item.current_round-1]
            assert len(item.model_response) == item.current_round
            # breakpoint()
            if model_response is None or item.status != "processing":
                continue
            tool_cfg = None
            try:
                # breakpoint()
                if self.model_mode == "general":

                    actions, action_parse_error = self.extract_actions(model_response)
                    # breakpoint() 
                    if actions is not None:
                        action = actions[0]
                        assert 'name' in action and 'arguments' in action, "missing 'name' or 'arguments' in the parsed action."
                        action_parse_error = None
                        tool_cfg = [{'API_name': action['name'],
                                    'API_params': action['arguments']}]   
                    else:
                        tool_cfg = None
                        
            except Exception as e:
                msg = f"Invalid action format."
                logger.info(f"{msg}: {e}")
                action_parse_error = msg
                tool_cfg = None
                
            item.action_parse_error.append(action_parse_error if item.current_round != 1 else None)
            item.tool_cfg.append(tool_cfg)

    ## Batch Inference
    def batch_inference(self,dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=lambda x: x[0]
        )

        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
        self.dataloader_iter = iter(self.dataloader)
        self.tp_model.eval()

        progress_bar = tqdm_rank0(len(self.dataloader), desc="Model Responding")

        if len(self.dataloader) == 0 and not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
            return
        self.manager.append_item_to_full(self.dataloader_iter, progress_bar=progress_bar)


        current_batch = self.manager.get_current_batch()
        self.tp_model.generate(current_batch)

        self.manager.update_item_status()
        while len(current_batch) > 0:
            try:
                # Inspect and yield output
                results = self.manager.pop_qualified_items()
                for res in results:
                    idx = res["meta_data"]["idx"]
                    self.dataset.store_results(dict(idx=idx,results=res))

                # Parse tool config and generate tool response
                self.batch_parse_tool_config()
                
                self.batch_get_tool_response()
                self.batch_tool_response_to_next_round_input()
                
                # Refill the current batch
                self.manager.append_item_to_full(self.dataloader_iter,progress_bar=progress_bar)
                
                current_batch = self.manager.get_current_batch()
                self.tp_model.generate(current_batch)
                self.manager.update_item_status()

            except StopIteration:
                break
        # breakpoint()
        assert len(self.manager.get_current_batch()) == 0
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
    
    def convert_conv_to_msges(self, conv):
        messages = []
        for item in conv:
            if item["from"] == "human":
                clean_text = item["value"].replace('<AGENT_PROMPT>\n\nQuestion:', '').strip()
                self.append_conversation_fn( messages, text = clean_text, role = "user")
            if item["from"] == "gpt":
                self.append_conversation_fn(messages, text = item["value"] ,role = "assistant") 
        return messages
    
    def step_inference(self,dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=lambda x: x[0],  
            shuffle=False,
        )

        if dist.is_initialized() and not 'vllm_models' in str(type(self.tp_model)):
            self.dataloader = self.accelerator.prepare(self.dataloader)
        self.tp_model.eval()

        if len(self.dataloader) == 0 and not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
            return
        
        progress_bar = tqdm_rank0(len(self.dataloader), desc="Model Responding (step-mode)")

        for batch in self.dataloader:
            if progress_bar is not None:
                progress_bar.update(1)
                
            batch["input"] = self.convert_conv_to_msges(batch["text"])
            result = {}
            result["meta_data"] = copy.deepcopy(batch)
            result["model_output"] = self.tp_model.generate(batch["input"])
            self.dataset.store_results(dict(idx=batch["idx"],results=result))
        
        if not 'vllm_models' in str(type(self.tp_model)):
            self.accelerator.wait_for_everyone()
    
    
