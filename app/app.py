import gradio as gr
import json
import re
import torch
import os
from PIL import Image
import rasterio
import numpy as np
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "app", "gradio_tmp")

# ---------------------------------------------------------
# Model Class (same as yours)
# ---------------------------------------------------------
class LLM:
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
# Extract tool actions
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
    
# ---------------------------------------------------------
# Convert .tif to PNG preview
# ---------------------------------------------------------
def make_preview_if_tif(path):
    if path and path.lower().endswith(".tif"):
        png_path = path.replace(".tif", "_preview.png")
        with rasterio.open(path) as src:
            data = src.read([1, 2, 3])
            data = np.transpose(data, (1, 2, 0))
            img = Image.fromarray(data)
            img.save(png_path)  

        return png_path
    return path
def handle_upload(image_path):
    if not image_path:
        return None, None
    if image_path.lower().endswith(".tif"):
        preview_path = make_preview_if_tif(image_path)
        return preview_path, image_path
    return image_path, None

# ---------------------------------------------------------
# Load model + tools ONCE (important)
# ---------------------------------------------------------
PRETRAINED_PATH = "MBZUAI/OpenEarthAgent"

Model = LLM(PRETRAINED_PATH)
tool_manager = ToolManager()

IMAGE_REQUIRED_TOOLS = [ "OCR", "DrawBox", "AddText", "TextToBbox","CountGivenObject",
                        "ImageDescription", "RegionAttributeDescription","ChangeDetection",
                        "SegmentObjectPixels","ObjectDetection","GetBboxFromGeotiff","DisplayOnGeotiff"
                    ]
GPKG_REQUIRED_TOOLS = [ "AddPoisLayer","ComputeDistance","DisplayOnMap","AddIndexLayer",
                        "ComputeIndexChange","ShowIndexLayer","DisplayOnGeotiff"
                    ]

# ---------------------------------------------------------
# Chat Function for Gradio
# ---------------------------------------------------------
def run_agent(user_question, input_image, original_tif):

    logs = ""
    conversation = [{"role": "user", "content": user_question}]
    current_image = None
    current_gpkg = None
    if original_tif:
        input_image = original_tif

    if input_image and os.path.isfile(input_image):
        if input_image.endswith("S_10_preview.png"):
            current_image = os.path.abspath("./assets/S_10.tif")  ## Hack for sample examples only
        else:
            current_image = input_image
        file_name = os.path.basename(current_image)
        logs += f"Loaded image: {file_name}\n\n"
        yield "", logs, None

    for round_id in range(1, 16):  # max tool rounds

        logs += f"\n--- ROUND {round_id} ---\n"
        yield "", logs, None
        model_output = Model.generate(conversation)
        logs += f"\nAGENT PLAN and ACTION:\n{model_output}\n"
        yield "", logs, None
        actions, err = extract_actions(model_output)
        if err:
            logs += "\nNo actions found.\n"
            yield "", logs, None
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": err + " " + user_question})
            continue
        if actions == []:
            if round_id != 1:
                logs += "\nNo actions found.\n"
                yield "", logs, None
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_question})
            continue

        action = actions[0]
        api_name = action["name"]
        api_args = action["arguments"]

        if api_name == "Terminate":
            final_answer = api_args.get("ans", "")
            logs += f"\nFINAL ANSWER:\n{final_answer}\n"
            if current_image:
                preview_image = make_preview_if_tif(current_image)
            if final_answer == "<image>":
                yield final_answer, logs, preview_image
            else:
                yield final_answer, logs, None
            return
        
        logs += f"\nCalling Tool: {api_name}\n"
        logs += f"Arguments: {json.dumps(api_args, indent=4)}\n"
        yield "", logs, None
        
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

        tool_resp = tool_manager.call_tool(api_name, api_args)
        if tool_resp.get("error_code") == 0:
            logs += f"✅ TOOL RESPONSE:\n"
            logs += f"{tool_resp.get('text')}\n"
            yield "", logs, None
        else:
            logs += f"❌ TOOL RESPONSE:\n"
            logs += f"{tool_resp.get('text')}\n"
            yield "", logs, None


        if "gpkg" in tool_resp:
            current_gpkg = tool_resp.get("gpkg")
        if "image" in tool_resp:
            current_image = tool_resp.get("image")
     
        tool_text = tool_resp.get("text", "")
        obs = (
            f"OBSERVATION:\n{api_name} outputs: {tool_text}\n"
            "Please summarize and answer the question."
        )

        conversation.append({"role": "assistant", "content": model_output})
        conversation.append({"role": "user", "content": obs})

    return "Max tool rounds reached.", logs, current_image

# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------
logo_path = "./assets/OpenEarthAgent_logo_transparent.png"

with gr.Blocks() as demo:
    original_tif_state = gr.State()
    with gr.Column():
        gr.Image(logo_path, show_label=False, container=False, height=100, buttons=[])
        gr.Markdown("""
        <div style="text-align:center;">
            <h2 style="margin-top:-5px; margin-bottom:5px;">
                A Unified Framework for Tool-Augmented Geospatial Agents
            </h2>
            <h1 style="margin-top:0px;">
                Demo (Live)
            </h1>
        </div>
        """)

    with gr.Row():

        # LEFT
        with gr.Column(scale=1):
            user_input = gr.Textbox(label="Enter your question")
            image_input = gr.Image(
                type="filepath",
                label="Upload Image",
                height=500,
                sources=["upload","clipboard"],
                buttons=[]
            )
            image_input.change(
                fn=handle_upload,
                inputs=image_input,
                outputs=[image_input, original_tif_state],
                trigger_mode="once"
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", scale=1)
                clear_btn = gr.Button("Clear", scale=1)
            gr.Markdown("### Sample Queries")
            gr.Examples(
                examples=[
                    ["Generate a preview from the NBR difference for Topanga State Park, Los Angeles, USA, covering December 2024 and February 2025.", None],
                    ["For the area within a 1000m radius of Tokyo Skytree, compute and map the travel distances between each kindergarten and the nearest police station.", None],
                    ["Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", "./assets/S_10_preview.png"],
                    ["Locate and estimate the distance between aircrafts in the scene. Assuming GSD 0.6 px/meter", "./assets/TG_P0009.png"],
                ],
                inputs=[user_input, image_input],
                preprocess=False,
            )
        # RIGHT
        with gr.Column(scale=1):
            final_answer = gr.Textbox(label="Final Answer")

            execution_trace = gr.Textbox(
                label="Execution Trace",
                lines=10,
                autoscroll=True
            )

            output_image = gr.Image(type="filepath", label="Output Image", height=500, buttons=["download"])

    submit_btn.click(
        run_agent,
        inputs=[user_input, image_input, original_tif_state],
        outputs=[final_answer, execution_trace, output_image]
    )

    clear_btn.click(
        lambda: ("", None, "", None),
        inputs=[],
        outputs=[user_input, image_input, execution_trace, output_image, final_answer]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=4444, allowed_paths=["./"])
