#!/usr/bin/env python3
import json
import re
import torch
import os
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenLM:
    def __init__(self, pretrained):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(pretrained,dtype=torch.bfloat16,trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained,use_fast=True,trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.system_prompt = RS_AGENT_PROMPT
        self.max_new_tokens = 256

    def prepend_system_prompt(self, conversation):
        if not conversation or conversation[0]["role"] != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation

    def generate(self, conversation):
        conversation = self.prepend_system_prompt(conversation)
        text = self.tokenizer.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
        inputs = self.tokenizer(text,return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs,max_new_tokens=self.max_new_tokens)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids,skip_special_tokens=True)

# ---------------------------------------------------------
# Extract tool actions from model output
# ---------------------------------------------------------
def extract_actions(text: str):
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
        return None, msg
    
IMAGE_REQUIRED_TOOLS = [ "OCR", "DrawBox", "AddText", "TextToBbox","CountGivenObject",
                        "ImageDescription", "RegionAttributeDescription","ChangeDetection",
                        "SegmentObjectPixels","ObjectDetection","GetBboxFromGeotiff","DisplayOnGeotiff"
                    ]
GPKG_REQUIRED_TOOLS = [ "AddPoisLayer","ComputeDistance","DisplayOnMap","AddIndexLayer",
                        "ComputeIndexChange","ShowIndexLayer","DisplayOnGeotiff"
                    ]
# ---------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------
def run_chat(pretrained_path, max_rounds=10):
    """
    Loads model and runs a tool-chat loop.
    pretrained_path: e.g. "MBZUAI/OpenEarthAgent"
    """

    # --------------------------------------
    # Load Model
    # --------------------------------------
    Model = QwenLM(pretrained_path)
    print(f"Loaded model:  pretrained={pretrained_path}")

    # --------------------------------------
    # Load tools
    # --------------------------------------
    tool_manager = ToolManager()
    available = tool_manager.available_tools
    print("\n🔧 Available tools:", available)

    # --------------------------------------
    # Initialize conversation
    # --------------------------------------
    conversation = []
    current_image = None
    current_gpkg = None
    user_input = input("\nUser question: ")
    input_image_path = input("\nInput image path: ")
    if input_image_path:
        if os.path.isfile(input_image_path):
            current_image = input_image_path
            print(f"Image loaded successfully: {current_image}")
        else:
            print("Invalid Image")
    conversation.append({"role": "user", "content": user_input})

    # --------------------------------------
    # Chat loop
    # --------------------------------------
    for r in range(1, max_rounds + 1):
        print(f"\n--- Round {r}/{max_rounds} ---")

        # 1) MODEL RESPONSE
        model_output = Model.generate(conversation)
        print("\n🤖 Model output:\n", model_output)

        # 2) PARSE ACTIONS
        actions, err = extract_actions(model_output)

        if err:
            print("No Action Parsed")
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": err + " " + user_input})
            continue
        if actions == []:
            print("Action Empty")
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_input})
            continue

        action = actions[0]
        api_name = action["name"]
        api_args = action["arguments"]

        # TERMINATE tool
        if api_name == "Terminate":
            print("\n🏁 Final Answer:")
            print(api_args.get("ans", ""))
            break
        
        if api_name in GPKG_REQUIRED_TOOLS:
            if "gpkg" in api_args:
                api_args['gpkg'] = current_gpkg
        if api_name in IMAGE_REQUIRED_TOOLS:
            if "image" in api_args:
                api_args["image"] = current_image
        if api_name in ["DisplayOnGeotiff", "GetBboxFromGeotiff"]:
            if "geotiff" in api_args:
                api_args["geotiff"] = current_image
        if api_name == "AddText":
            api_args["color"] = "green"
            
        print(f"\n🔧 Calling tool: {api_name} with args={api_args}")
        tool_resp = tool_manager.call_tool(api_name, api_args)

        if "gpkg" in tool_resp:
            current_gpkg = tool_resp.get("gpkg")
        if "image" in tool_resp:
            current_image = tool_resp.get("image")

        print("\n🛠 Tool response:")
        print(tool_resp)

        # 4) FEED TOOL RESULT BACK TO MODEL
        
        tool_text = tool_resp.get("text", "")
        obs = (
            f"OBSERVATION:\n{api_name} model outputs: {tool_text}\n"
            "Please summarize the model outputs and answer my first question."
        )

        conversation.append({"role": "assistant", "content": model_output})
        conversation.append({"role": "user", "content": obs})

    print("\n=== Chat finished ===")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example:
    # python chat.py
    run_chat(
        pretrained_path="MBZUAI/OpenEarthAgent",
        max_rounds=15
    )
