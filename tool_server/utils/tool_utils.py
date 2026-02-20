from typing import Any, List, Union
import re
import json
import numpy as np
import torch
import io
import base64
from PIL import Image
from typing import Dict, Optional
from .utils import pil_to_base64, base64_to_pil


class ImageToolManager:
    """
    Manager to handle image storage and conversion for tool calls.
    """
    def __init__(self):
        self.image_dict = {}
        self.current_img_index = 1

    def add_initial_image(self, image: Image.Image) -> str:
        """
        Convert the initial image to a base64 string and store it in the dictionary.
        
        Args:
            image (Image.Image): The initial PIL image.
            
        Returns:
            str: The key for the stored image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        self.image_dict['img_1'] = img_base64
        return 'img_1'

    def process_base64_image(self, base64_str: str) -> Optional[str]:
        """
        Decode a base64 string into a PIL image and store it in the dictionary.
        
        Args:
            base64_str (str): Base64 encoded image string.
        
        Returns:
            Optional[str]: The image key if processing succeeds; otherwise, None.
        """
        try:
            # Decode the base64 string into image data
            image_data = base64.b64decode(base64_str)
            # Convert the decoded bytes into a PIL image
            image = Image.open(io.BytesIO(image_data))
            
            # Increment the image index and create a new key
            self.current_img_index += 1
            new_img_key = f'img_{self.current_img_index}'
            
            # Store the image in the dictionary
            self.image_dict[new_img_key] = image
            return new_img_key
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None

    def store_tool_image(self, base64_str: str) -> Optional[str]:
        """
        Decode a base64 string into a PIL image and store it.
        (Note: This function does not currently return the new image key on success.)
        
        Args:
            base64_str (str): Base64 encoded image string.
        
        Returns:
            Optional[str]: The image key if processing succeeds; otherwise, None.
        """
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            
            self.current_img_index += 1
            new_img_key = f'img_{self.current_img_index}'
            self.image_dict[new_img_key] = image
            
            # Optionally, you might want to return new_img_key here.
            return new_img_key
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None

    def get_image_by_key(self, img_key: str) -> Optional[Image.Image]:
        """
        Retrieve an image from the storage by its key.
        
        Args:
            img_key (str): The key corresponding to the stored image.
            
        Returns:
            Optional[Image.Image]: The retrieved PIL image, or None if not found.
        """
        return self.image_dict.get(img_key)

def detect_tool_config(model_response: str, model_mode: str = "general") -> bool:
    """
    Detect whether the model response contains a tool call configuration.
    
    Args:
        model_response (str): The output string from the model.
        model_mode (str, optional): The mode for detection ('general' or 'llava_plus'). Defaults to "general".
        
    Returns:
        bool: True if a tool configuration is detected; otherwise, False.
    """
    if not model_response:
        return False

    if model_mode == "general":
        # Regex to match nested JSON-like structure
        pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
        match = re.search(pattern, model_response)
        if match:
            content = match.group(0)
            return '"actions"' in content
        return False

    elif model_mode == "llava_plus":
        # Pattern specific to the "llava_plus" mode
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        return bool(re.search(pattern, model_response, re.DOTALL))

    return False


def parse_tool_config(
    model_response: str, 
    model_mode: str = "general", 
    image_tool_manager: Optional[ImageToolManager] = None,
    newest_image: Optional[Image.Image] = None
) -> Optional[List[Dict]]:
    """
    Parse the tool configuration from the model response and handle image conversion if necessary.
    
    Args:
        model_response (str): The model's generated response.
        model_mode (str): The mode for parsing.
        image_tool_manager (Optional[ImageToolManager]): Manager for image conversion and storage.
        
    Returns:
        Optional[List[Dict]]: A list of parsed tool configurations, or None if parsing fails.
    """
    
    def extract_actions(text: str):
        """
        Extract only the 'actions' list from the model response text.
        
        Args:
            text (str): The model response text containing actions
            
        Returns:
            Optional[List]: The parsed actions list or None if extraction fails
        """
        try:
            # Try to find the "actions" part using regex
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
    
    if not model_response:
        return None

    try:
        if model_mode == "general":
            # pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
            # match = re.search(pattern, model_response)
            # if not match:
            #     return None
            # data = json.loads(match.group(0))

            # if "actions" not in data or not data["actions"]:
            #     return None
            actions = extract_actions(model_response)
            if not actions:
                return None

            parsed_actions = []
            for action in actions:
                # If an image (as base64) is provided in the action's arguments, process it.
                if image_tool_manager and 'image' in action.get('arguments', {}):
                    base64_img = action['arguments']['image']
                    img_key = image_tool_manager.process_base64_image(base64_img)
                    if img_key:
                        action['arguments']['image'] = img_key
                elif newest_image and 'image' in action.get('arguments', {}):
                    newest_image_base64 = pil_to_base64(newest_image, url_format=False)
                    action['arguments']['image'] = newest_image_base64

                parsed_actions.append({
                    "API_name": action["name"],
                    "API_params": action["arguments"]
                })

            return parsed_actions

        elif model_mode == "llava_plus":
            pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
            matches = re.findall(pattern, model_response, re.DOTALL)
            if not matches:
                return None
            # Extract the actions part and load it as JSON (ensure valid quotes)
            actions_str = matches[0][1].strip()
            return json.loads(actions_str.replace("'", "\""))

    except Exception as e:
        print(f"[parse_tool_config] Error: {e}")
        print("Wrong model response:", model_response)
        return None

def handle_tool_result(
    cfg,
    tool_result,
    conversations,
    model_mode: str = "general",
    original_prompt: Optional[str] = None,
    input_data_item: Dict = None,
):
    """
    Process the tool result, update the conversation history, and generate a new prompt.
    
    Args:
        tool_result: The result returned by the tool call.
        conversations: The current conversation history.
        model_mode (str): The mode for generating the updated prompt.
        original_prompt (Optional[str]): The original user prompt.
        image_tool_manager (Optional[ImageToolManager]): Manager for handling image conversions.
        
    Returns:
        The updated conversation history.
    """
    edited_image = None
    new_round_prompt = original_prompt

    if tool_result is not None:
        try:
            # Process image editing if an "edited_image" is provided in the tool result.
            if "edited_image" in tool_result:
                # Remove the edited image from the result and add it via the image manager.
                edited_image = tool_result.pop("edited_image")
                # # Note: Assumes that image_tool_manager has an 'add_image' method.
                # image_tool_manager.add_image(edited_image)
                # Convert the base64 string to a PIL image.
                # if edited_image == None:
                #     breakpoint()
                edited_image = base64_to_pil(edited_image)
                if input_data_item:
                    input_data_item["images"].append(edited_image)
                    
            else:
                edited_image = None

            # Extract text output from the tool result.
            tool_response_text = tool_result.get("text", None)
            # Retrieve the API name from the result (supporting multiple key names)
            api_name = cfg.get("API_name", cfg.get("api_name", ""))

            # Construct a new prompt based on the model mode.
            if model_mode == "llava_plus": 
                new_response = f"{api_name} model outputs: {tool_response_text}\n\n"
                new_round_prompt = (
                    f"{new_response} Please summarize the model outputs "
                    f"and answer my first question."
                )
            elif model_mode == "general":
                new_response = f"OBSERVATION:\n{api_name} model outputs: {tool_response_text}\n"
                new_round_prompt = (
                    f"{new_response}Please summarize the model outputs "
                    f"and answer my first question."
                )

        except Exception as e:
            # In case of errors, revert to the original prompt.
            print(f"Error in handle_tool_result: {e}")
            edited_image = None
            new_round_prompt = original_prompt

    # Pop previous images since vllm only supports one image
    # if input_data_item:
    #     for conv in input_data_item["conversations"]:
    #         for idx,c in enumerate(conv["content"]):
    #             if c["type"] == "image" or c["type"] == "image_url":
    #                 del conv["content"][idx]
    # Append the new message (with text and optional image) to the conversation history.
    updated_conversations = append_conversation_fn(
        conversation=conversations, 
        text=new_round_prompt, 
        image=input_data_item["images"][-1] if input_data_item else edited_image, 
        role="user"
    )

    return updated_conversations


def generate_conversation_fn(
    model,
    current_prompt_inputs: dict,
    tokenizer,
    generation_config
):
    """
    Generate a small step in the conversation. Returns the new text generated and the full output tensor.
    
    Args:
        model: The generation model.
        current_prompt_inputs (dict): The current prompt inputs.
        tokenizer: The tokenizer for decoding generated tokens.
        generation_config: Generation configuration parameters.
        
    Returns:
        Tuple containing the newly generated text and the full output tensor.
    """
    full_outputs = model.generate(
        **current_prompt_inputs,
        **generation_config.to_dict()
    )

    # Calculate the number of new tokens generated
    old_length = current_prompt_inputs["input_ids"].shape[1]
    new_tokens = full_outputs[:, old_length:]

    # Decode the new tokens into text
    new_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return new_text, full_outputs


def append_conversation_fn(
    conversation, 
    text: str, 
    image=None, 
    role: str = "user",
    # formatting: str = "qwen"
):
    """
    Append a new message to the conversation history.
    
    Args:
        conversation (list): The current conversation history.
        text (str): The text message to append.
        image: (Optional) An image to include with the message.
        role (str): The role of the sender (default is "user").
        
    Returns:
        The updated conversation list.
    """

    if image:
        image_base64 = pil_to_base64(image, url_format=True)
        new_messages = [
            {
                "role": role,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    else:
        new_messages = [
            {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    
    conversation.extend(new_messages)
    return conversation

