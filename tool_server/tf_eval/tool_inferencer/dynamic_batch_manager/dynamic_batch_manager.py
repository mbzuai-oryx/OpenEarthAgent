from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List
from tool_server.tf_eval.utils.log_utils import get_logger
from ...utils.utils import *
from PIL import Image
import re
import json
logger = get_logger(__name__)

@dataclass
class DynamicBatchItem:
    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    meta_data: Dict = field(default = None)
    conversation: object = field(default = None)
    model_response: List[str] = field(default_factory=list)
    tool_cfg :  List[str] = field(default_factory=list)
    tool_response :  List[str] = field(default_factory=list)
    action_parse_error : List[str] = field(default_factory=list)
    new_round_input :  List[str] = field(default_factory=list)
    gpkg : str = field(default_factory=str)
    img : str = field(default_factory=str)
    cached_gpkg : str = field(default_factory=str)
    cached_layers: Dict[str, str] = field(default_factory=dict)

class DynamicBatchManager():
    def __init__(
        self,
        batch_size: int,
        stop_token: str = "<stop>",
        max_rounds: int = 3,
        generate_conversation_fn = None,
    ):
        self.dynamic_batch = []
        self.batch_size = batch_size
        self.stop_token = stop_token
        self.max_rounds = max_rounds
        self.generate_conversation_fn = generate_conversation_fn
    
    def extract_action(self, text: str):
        try:
            actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
            actions_match = re.search(actions_pattern, text)
            if not actions_match:
                return None
            actions_str = actions_match.group(1)
            actions_list = json.loads(actions_str)
            return actions_list
            
        except Exception as e:
            print(f"Error extracting actions list: {e}")
            return None
        
    def extract_final_answer(self, final_response: str):
        res = ""
        actions = self.extract_action(final_response)
        if actions:
            action = actions[0]
            if not isinstance(action, dict):
                return res
            if (
                action.get("name") == "Terminate"
                and isinstance(action.get("arguments"), dict)
                and "ans" in action["arguments"]
            ):
                ans = action["arguments"]["ans"]
                return ans.strip() if isinstance(ans, str) else ans
        return res
    
    def check_terminate(self, response: str):
        actions = self.extract_action(response)
        if actions:
            action = actions[0]
            if 'name' in action and 'arguments' in action:
                if action["name"] == "Terminate":
                    return True
        return False
    
    def pop_qualified_items(self):
        res = []
        new_batch = []
        for idx,item in enumerate(self.dynamic_batch):
            if item.status == "finished":
                item = asdict(item)       
                final_model_output = item["model_response"][-1]
                final_answer = self.extract_final_answer(final_model_output)
                item["final_answer"] = final_answer
                
                res.append(item)
            else:
                new_batch.append(item)
        self.dynamic_batch = new_batch
        return res
    
    def append_item(self, meta_data: Dict):
        if len(self.dynamic_batch) < self.batch_size:
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
            )
            candidate_item.conversation = self.generate_conversation_fn(
                text = meta_data["text"], 
                role = "user"
            )
            
            self.dynamic_batch.append(candidate_item)
        else:
            raise ValueError("Batch is full")
    
    def append_item_to_full(self, dataloader, progress_bar=None):
        while len(self.dynamic_batch) < self.batch_size:
            try:
                self.append_item(next(dataloader))
                if progress_bar:
                    progress_bar.update(1)
            except:
                break
        
    def get_current_batch(self):
        return self.dynamic_batch
    
    # Caution: Call this only after model.generate
    def update_item_status(self):
        for item in self.dynamic_batch:
            if item.status == "pending":
                if item.current_round == item.max_rounds or self.check_terminate(item.model_response[-1]):
                    item.status = "finished"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                if item.current_round == item.max_rounds or self.check_terminate(item.model_response[-1]):
                    item.status = "finished"
                else:
                    item.current_round += 1
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
        
    
    