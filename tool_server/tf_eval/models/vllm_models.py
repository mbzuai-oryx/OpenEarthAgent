from .abstract_model import tp_model
import uuid
from PIL import Image
from typing import List
from vllm import LLM, SamplingParams
from vllm import TokensPrompt
from .template_instruct import *
from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.rs_agent_prompt import RS_AGENT_PROMPT

from ..utils.log_utils import get_logger
inferencer_id = str(uuid.uuid4())[:6]
logger = get_logger("vllm_models",)

class VllmModels(tp_model):
    def __init__(
      self,  
      pretrained : str = None,
      tensor_parallel: str = "1",
      limit_mm_per_prompt: str = "0"
    ):
        self.max_new_tokens = None
        self.default_max_new_tokens = 256
        self.max_model_len = 8192
        tensor_parallel = eval(tensor_parallel)
        self.model = LLM(
            model=pretrained,
            tensor_parallel_size=tensor_parallel,
            limit_mm_per_prompt={"image": int(limit_mm_per_prompt)},
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len= self.max_model_len,
            dtype="bfloat16",
            tokenizer_mode='auto'
        )
        self.system_prompt = RS_AGENT_PROMPT
        logger.info(
                f"Initialized VllmModels with model={pretrained}, "
                f"tensor_parallel={tensor_parallel}, "
                f"max_model_len={self.max_model_len}, "
                f"limit_mm_per_prompt={limit_mm_per_prompt}, "
            )

    def generate_conversation_fn(self, text, role="user"):
        return [{"role": role, "content": text}]

    def append_conversation_fn(self, conversation, text, role):
        conversation.append({"role": role, "content": text})
        return conversation
    
    def prepend_system_prompt(self, conversation):
        """Ensure a system prompt is always the first message."""
        if not conversation or conversation[0].get("role") != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation
    
    def getitem_fn(self, meta_data, idx):
        res = meta_data[idx]
        return res

    def prepare_inputs(self, messages):
        tokenizer = self.model.get_tokenizer()
        SAFETY_MARGIN = 64
        max_model_len = self.max_model_len
        max_new_tokens = self.max_new_tokens or self.default_max_new_tokens
        limit = max_model_len - max_new_tokens - SAFETY_MARGIN
        if limit < 256: 
            raise ValueError("Wrong Limit")
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        msgs = messages.copy()     
        while len(input_ids) > limit and len(msgs) > 4:
            msgs.pop(2)  # drop oldest (keep system, question)
            msgs.pop(2) # remove turn
            suffix = "\n<previous context truncated>"
            if suffix not in msgs[1]["content"]:
                msgs[1]["content"] += suffix
            logger.warning(f"Prompt too long ({len(input_ids)}). Truncating to {limit}.")
            input_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        return input_ids

    def form_input_from_dynamic_batch(self, batch: List[DynamicBatchItem]):
        if len(batch) == 0:
            return None
        prompts = []
        for item in batch:
            conv = self.prepend_system_prompt(item.conversation)
            input_ids = self.prepare_inputs(conv)
            prompts.append(TokensPrompt(prompt_token_ids=input_ids))
        
        return prompts
    
    def form_input_from_msg(self, messages):
        conv = self.prepend_system_prompt(messages)
        input_ids = self.prepare_inputs(conv)
        return [TokensPrompt(prompt_token_ids=input_ids)]

    def generate(self, batch):
        if not batch or len(batch) == 0:
            return
        self.max_new_tokens = self.max_new_tokens if self.max_new_tokens is not None else self.default_max_new_tokens
        is_dynamic_batch = all(isinstance(x, DynamicBatchItem) for x in batch)
        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.6)

        if is_dynamic_batch:
            inputs = self.form_input_from_dynamic_batch(batch)
        else:
            inputs = self.form_input_from_msg(batch)

        response = self.model.generate(inputs, sampling_params)

        if is_dynamic_batch:
            for item, output_item in zip(batch, response):
                output_text = output_item.outputs[0].text
                item.model_response.append(output_text)
                self.append_conversation_fn(
                    item.conversation, output_text, "assistant"
                )
        return [response[0].outputs[0].text]

    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self
            
        
        
