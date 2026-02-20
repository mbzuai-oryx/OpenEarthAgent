
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
import re
import json
from datasets import Dataset
from collections import Counter

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not provided or missing in environment variable OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

LOG_DIR = "logs/scores"
os.makedirs(LOG_DIR, exist_ok=True)

TOOLS_LIST = ["Calculator","OCR","DrawBox","AddText","GoogleSearch","Plot","Solver",
    "TextToBbox","ImageDescription","RegionAttributeDescription","CountGivenObject",
    "ChangeDetection","SegmentObjectPixels","ObjectDetection",
    "GetAreaBoundary","AddPoisLayer","ComputeDistance","DisplayOnMap",
    "AddIndexLayer","ComputeIndexChange","ShowIndexLayer",
    "GetBboxFromGeotiff","DisplayOnGeotiff","Terminate"
]

def extract_action(text: str):
    try:
        actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
        actions_match = re.search(actions_pattern, text)
        if not actions_match:
            return None
        return json.loads(actions_match.group(1))     
    except Exception as e:
        # logger.info(f"Error extracting actions list: {e}")
        return None
        
def extract_final_answer(final_response: str):
    res = ""
    actions = extract_action(final_response)
    if actions:
        action = actions[0]
        if (
            action.get("name") == "Terminate"
            and isinstance(action.get("arguments"), dict)
            and "ans" in action["arguments"]
        ):
            ans = action["arguments"]["ans"]
            return ans.strip() if isinstance(ans, str) else ans
    return res

def extract_tool_cfg(action):
    tool_cfg = None
    error = ""
    if not isinstance(action, dict):
        error = "action_is_not_valid_dict"
        return tool_cfg, error
    if "name" not in action:
        error = "missing_name_key"
        return tool_cfg, error
    if "arguments" not in action:
        error = "missing_arguments_key"
        return tool_cfg, error
    if not isinstance(action["arguments"], dict):
        error = "arguments_is_not_valid_dict"
        return tool_cfg, error
    tool_cfg = {
            "API_name": action["name"],
            "API_params": action["arguments"],
        }
    return tool_cfg, error

def get_all_cfg_from_conversation(conversation):
    tool_cfg = []
    for conv in conversation:
        if conv.get('from') == 'gpt':
            msg = conv.get('value')
            action = extract_action(msg)
            if action:
                cfg, _ = extract_tool_cfg(action[0])
                if cfg:
                    tool_cfg.append(cfg)
    return tool_cfg

def last_tool_in_gen(conversations):
    gen_tools = ["DrawBox","AddText","Plot","DisplayOnMap","ShowIndexLayer","DisplayOnGeotiff"]
    cfgs = get_all_cfg_from_conversation(conversations)
    tool_names = [cfg["API_name"] for cfg in cfgs if cfg["API_name"] != "Terminate"]
    last_tool = tool_names[-1]
    return last_tool in gen_tools

EVAL_PROMPT = """
You are an evaluation assistant responsible for measuring the Answer Accuracy Score (a value between 0 and 1) for a geospatial agent’s prediction.

You are given:

- A Question  
- A Ground Truth Answer  
- A Predicted Answer produced by the agent  

Your task is to evaluate how accurately the Predicted Answer matches the Ground Truth Answer, considering both numerical precision and semantic correctness.  
When numerical values are present, allow a ±10% tolerance range for acceptable variation. For descriptive or categorical outputs, judge based on meaningful semantic equivalence.

Evaluation Guidelines

1. Numerical or Quantitative Comparisons  
- A predicted numeric value is correct if it lies within ±10% of the corresponding value in the ground truth.  
- For bounding boxes, consider the prediction correct if the IoU (Intersection over Union) ≥ 0.5.
- If multiple numeric values exist:  
    - Compute element-wise difference.
    - All values within ±10% range → Score = 0.95 – 1.00 (highly accurate)  
    - Most values within ±10% range → Score = 0.80 – 0.94 (minor acceptable deviations)  
    - Some values within ±10% range → Score = 0.30 – 0.79 (partially correct)  
    - Most values within ±20% range → Score = 0.10 – 0.29 (partially aligned)  
    - All values incorrect or nonsensical → Score = 0.00 (completely incorrect)  
- Ignore extraneous numbers that do not directly answer the question.

2. Non-Numerical or Categorical Comparisons  
    - Evaluate semantic equivalence:  
    - Perfect semantic match → Score ≈ 1.0  
    - Minor phrasing or synonym difference → 0.8–0.9  
    - Partially correct (missing non-critical details) → 0.4–0.8  
    - Vague or weakly related → 0.1–0.4  
    - Contradictory or incorrect → 0.0

3. Handling Missing or Contradictory Content  
- If the prediction omits all required answer elements → Score = 0.0
- If the prediction introduces fabricated entities, relationships, or placeholders → Score = 0.00
- If a numeric answer is required and the model responds with “no data”, “unknown”, or similar → Score = 0.00

You must provide your evaluation strictly in the following format and nothing else:
{
    "Score": <float between 0.0 and 1.0>,
    "Justification": "<1–2 concise sentences explaining the score based on the comparison between GT and Pred>"
}
"""
def extract_score(eval_text: str):
    before_just = eval_text.split("Justification")[0]
    match = re.search(r"[-+]?\d*\.\d+|\d+", before_just)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0

def sim_score(ans_gold: str,ans_pred: str, question: str):
    if not ans_pred:
        return 0.0
    
    prompt = f"{EVAL_PROMPT}\n\nQuestion: {question}\nGround Truth Answer: {ans_gold}\nPredicted Answer: {ans_pred}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    evaluation = response.choices[0].message.content.strip()
    score = extract_score(evaluation)
    log_path = os.path.join(LOG_DIR, f"LLM_scores.jsonl")
    entry = {
        "question": question,
        "ground_truth": ans_gold,
        "predicted": ans_pred,
        "evaluation": evaluation,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return score
    

def gettype(name: str):
    perception = ['OCR', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox', 'ChangeDetection', 'ObjectDetection', 'SegmentObjectPixels', 'CountGivenObject']
    operation = ['DrawBox', 'AddText', 'GoogleSearch']
    logic = ['Calculator', 'Solver', 'Plot']  
    gis = ["GetAreaBoundary", "AddPoisLayer", "ComputeDistance", "DisplayOnMap", "AddIndexLayer", "ComputeIndexChange", "ShowIndexLayer", 'GetBboxFromGeotiff', "DisplayOnGeotiff"]
    terminate = ["Terminate"]
    if name in perception:
        return 'perception'
    elif name in operation:
        return 'operation'
    elif name in logic:
        return 'logic'
    elif name in gis:
        return 'gis'
    elif name in terminate:
        return 'terminate'
    else:
        return 'none'

def load_dataset(file_path, num_samples=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if num_samples:
        dataset = dataset[:num_samples]
    return Dataset.from_list(dataset)

def load_data_function():
    
    dataset_path = task_config["dataset_path"]
    image_dir_path = task_config["image_dir_path"]
    num_samples = task_config['num_sample']
    task_type = task_config['task_type'] # e2e or step
    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    step_idx = 0
    for index , item in enumerate(dataset):

        idx_val = item.pop("idx")
        idx = f"rsagent_{index}"
        images = item.get("images",[])
        image_paths = [
            os.path.join(image_dir_path, os.path.basename(img)) if not os.path.isabs(img) else img
            for img in images
        ]
        item.pop("images")
        if task_type == "e2e":
            text = item["conversation"][0]["value"].split("\n\nQuestion:", 1)[1].strip() # text = question
            label = item.pop("label", None)
            if label is None:
                try:
                    last_value = item["conversation"][-1].get("value", "")
                    label = extract_final_answer(last_value)
                except Exception as e:
                    logger.warning(f"Failed to extract label for idx={idx_val}: {e}")
                    label = ""
            data_item = dict(idx=idx, sample_num=idx_val, text=text, label=label, images=image_paths, **item)
            meta_data.append(data_item)

        elif task_type == "step":
            conversation = item["conversation"]
            conv_num = 0
            for i, entry in enumerate(conversation):
                if entry["from"] == "human":
                    text = conversation[: i + 1] # text = conversation upto i
                    label = conversation[i + 1]["value"] # label = next response
                    conv_num += 1
                    step_idx += 1 
                    data_item = dict(idx=f"rsagent_{step_idx}", sample_num=idx_val, conv_num=conv_num, text=text, label=label, images=image_paths, **item)
                    meta_data.append(data_item)
        else:
            raise ValueError(f"Unsupported task type: '{task_type}'")
        
    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data

def compare_tool_sequences(API_names_gt, API_names_pred):
    """
    Compare predicted vs ground-truth tool call sequences.

    Returns:
      same_order: GT sequence appears as subsequence in pred (extras allowed)
      any_order:  All GT tools appear in pred with >= same counts
      unique:     All unique GT tools appear in pred (order ignored)
    """
    # --- multiset inclusion (any order, with counts) ---
    gt_counter, pred_counter = Counter(API_names_gt), Counter(API_names_pred)
    any_order = all(pred_counter[k] >= v for k, v in gt_counter.items())

    # --- order check (GT as subsequence of pred) ---
    same_order = False
    if any_order:  # Only check order if all GT tools appear
        i = 0
        for name in API_names_pred:
            if i < len(API_names_gt) and name == API_names_gt[i]:
                i += 1
        same_order = i == len(API_names_gt)

    # --- unique version (ignoring repeats) ---
    def unique(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    gt_unique, pred_unique = unique(API_names_gt), unique(API_names_pred)
    unique_ok = all(t in pred_unique for t in gt_unique)

    return same_order, any_order, unique_ok

def evaluate_function(results,meta_data):

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    task_type = task_config['task_type'] 
    metrics = {}
    error_breakdown = {}

    if task_type == "e2e":
        metrics = {
            'perception_f1': 0, 
            'operation_f1': 0, 
            'logic_f1': 0,
            'gis_f1' : 0,
            'total_tool_calls_gt' : 0,
            'total_actions_parsed': 0,
            'total_tool_calls_pred' : 0,
            'tool_call_error': 0,
            'answer_acc' : 0,
            'answer_acc_w_gen' : 0,
            'all_tools_called_any_order' : 0,  # preserves multiple instances of tool calls
            'all_tools_called_same_order' : 0,
            'all_tools_called_unique' : 0,     # check unique tool calls only 
            'avg_steps_taken': 0,
            'total_gen_samples': 0
        }
        error_breakdown = {
            'ans_not_reached' : 0,
            'name_not_in_tools': 0,
            'no_valid_action_taken' : 0,
            'no_action_found': 0,
            'action_found_but_wrong_syntax': 0,
            'answer_reached_without_tool_calls' : 0,
            'Multiple_tool_calls_in_single_step' : 0
        }

        gen_tools = [
            'DrawBox', 
            'AddText', 
            'Plot', 
            'DisplayOnMap',
            'ShowIndexLayer',
            "DisplayOnGeotiff"
            ]

        total_gt_calls_per_tool = { tool: 0 for tool in TOOLS_LIST}
        total_pred_calls_per_tool = {tool: {"called": 0, "success": 0} for tool in TOOLS_LIST}

        total = {'all': 0, 'all_gen': 0, 'answer': 0, 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        total_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        correct_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        precision = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        recall = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        f1 = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}

        for idx, meta in meta_dict.items():
            if idx in results_dict:
                try:
                    final_ans = results_dict[idx]["results"].get("final_answer")
                    if final_ans is not None:
                        meta["prediction"] = final_ans
                    else:
                        meta["prediction"] = None
                except Exception as e:
                    print(f"[Error] {e} in idx: {idx}")
                    meta["prediction"] = None
                tool_cfg_pred = results_dict[idx]["results"]["tool_cfg"] # all tool calls
                if not meta["prediction"]:
                    error_breakdown['ans_not_reached'] +=1
            else:
                logger.info(f"no prediction found for sample number {idx}")
                meta["prediction"] = "None"
            
            gold = meta["label"]
            pred = meta["prediction"]
            question = meta["conversation"][0]["value"].split("\n\nQuestion:", 1)[1].strip()

            metrics["avg_steps_taken"] += results_dict[idx]["results"]["current_round"] if idx in results_dict else 0
            tool_cfg_gt = get_all_cfg_from_conversation(meta["conversation"])

            API_names_gt = [cfg["API_name"] for cfg in tool_cfg_gt if cfg is not None]
            API_names_pred = [cfg[0]["API_name"] for cfg in tool_cfg_pred if cfg and isinstance(cfg[0], dict) and "API_name" in cfg[0]]

            tool_response = results_dict[idx]["results"]["tool_response"]
            error_codes = [res['error_code'] for res in tool_response if res]

            if API_names_gt[-2] in gen_tools: #if last tool before terminate is generation tool then calc gen_acc
                total['all_gen'] += 1
                # same last tool call and tool call was successful
                if len(API_names_pred) > 0 and API_names_gt[-2] == API_names_pred[-1] and tool_response[-1] and tool_response[-1]['error_code'] == 0: 
                    gen_score = 1  
                    metrics["answer_acc_w_gen"] += gen_score
            else:
                total['all'] += 1
                metrics["answer_acc"] += sim_score(gold, pred, question)

            metrics['total_tool_calls_gt'] += len(API_names_gt)
            metrics['total_tool_calls_pred'] += len(API_names_pred)
            same_order, any_order, unique_ok= compare_tool_sequences(API_names_gt[:-1], API_names_pred)
            if same_order:
                metrics['all_tools_called_same_order'] += 1
            if any_order:
                metrics['all_tools_called_any_order'] += 1
            if unique_ok:  
                metrics['all_tools_called_unique'] += 1

            for name in API_names_gt:
                if name in total_gt_calls_per_tool:
                    total_gt_calls_per_tool[name] += 1
            for name in API_names_pred:
                if name in total_pred_calls_per_tool:
                    total_pred_calls_per_tool[name]["called"]  += 1
            if (meta["prediction"]):
                total_pred_calls_per_tool["Terminate"]["called"]  += 1

            if len(API_names_pred) == len(error_codes):
                for name, code in zip(API_names_pred, error_codes):
                    if name in total_pred_calls_per_tool and code == 0:
                        total_pred_calls_per_tool[name]["success"] += 1

            metrics['tool_call_error'] += sum(code != 0 for code in error_codes) 
            for name in API_names_pred:
                tool_type = gettype(name) 
                if tool_type != "none":
                    total_predict[tool_type] += 1
                else:
                    error_breakdown['name_not_in_tools'] += 1 
            API_names_pred_copy = API_names_pred.copy()
            for name in API_names_gt:
                total[gettype(name)] += 1
                if name in API_names_pred_copy:  # any order
                    API_names_pred_copy.remove(name)
                    correct_predict[gettype(name)] += 1
            
            model_response = results_dict[idx]["results"]["model_response"] 
            # model should call tool in every response except for first response (can be thought only)
            error_breakdown['no_valid_action_taken'] += len(model_response)-len(tool_cfg_pred) - 1
            if meta["prediction"]: # if ans is reached
                if all(x is None for x in tool_cfg_pred): # no valid tool calls 
                    error_breakdown['answer_reached_without_tool_calls'] += 1

            for response in model_response:
                actions = extract_action(response)
                if actions and len(actions)>1:
                    error_breakdown['Multiple_tool_calls_in_single_step'] += 1

            action_parse_error = results_dict[idx]["results"]["action_parse_error"]
            for entry in action_parse_error:
                if not entry:
                    metrics["total_actions_parsed"] += 1
                else:
                    if "Invalid action format." in entry:
                        error_breakdown['action_found_but_wrong_syntax'] += 1
                    elif "No action found." in entry:
                        error_breakdown['no_action_found'] += 1
                    else:
                        logger.info(f"found unknown error: {entry}")

        for tool_type in f1.keys():
            precision[tool_type] = correct_predict[tool_type] / (total_predict[tool_type] + 1e-5)
            recall[tool_type]    = correct_predict[tool_type] / (total[tool_type] + 1e-5)
            f1[tool_type] = 2 * precision[tool_type] * recall[tool_type] / (precision[tool_type] + recall[tool_type] + 1e-5)

        metrics["answer_acc"] = round(metrics["answer_acc"] *100/ total['all'], 4) if total['all'] else 0.0
        metrics["answer_acc_w_gen"] = round(metrics["answer_acc_w_gen"] *100 / total['all_gen'], 4) if total['all_gen'] else 0.0
        metrics["perception_f1"] = f1['perception'] * 100
        metrics["operation_f1"] = f1['operation'] * 100
        metrics["logic_f1"] = f1['logic'] * 100
        metrics["gis_f1"] = f1['gis'] * 100
        metrics["avg_steps_taken"] = metrics["avg_steps_taken"] / (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        metrics["total_gen_samples"] = total['all_gen']
        metrics['all_tools_called_same_order'] /= (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        metrics['all_tools_called_any_order'] /= (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        metrics['all_tools_called_unique'] /= (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        return {"metrics":metrics, "error_breakdown": error_breakdown, "total_gt_calls_per_tool": total_gt_calls_per_tool, "total_pred_calls_per_tool": total_pred_calls_per_tool}
    
    elif task_type == "step":
        metrics = {
                'inst_acc': 0,
                'tool_acc': 0,
                'arg_acc1' : 0,
                'arg_acc2': 0,
                'answer_acc': 0,
                'count_non_gen_samples': 0,
                'count_gen_samples': 0
            }
        error_breakdown = {
            "no_valid_action" : 0,
            "action_is_not_valid_dict" : 0,
            "missing_name_key" : 0,
            "name_not_in_tools": 0,
            "incorrect_tool_called" : 0,
            "missing_arguments_key" : 0,
            "required_argument_missing" : 0,
            "arguments_is_not_valid_dict": 0,
            "argument_values_not_matching": 0
        }
        total_calls = 0
        count_non_gen_samples = 0
        count_gen_samples = 0
        unique_samples = len(set(meta.get("sample_num") for meta in meta_data if "sample_num" in meta))
        logger.info(f"Total unique samples: {unique_samples}")

        for idx, meta in meta_dict.items():
            if idx in results_dict:
                meta["prediction"] = results_dict[idx]["results"]["model_output"][0]
                actions= extract_action(meta["prediction"])
                if actions:
                    action = actions[0] 
                    tool_cfg_pred, error = extract_tool_cfg(action)
                else:
                    error = "no_valid_action"
                    tool_cfg_pred = None
            else:
                logger.info(f"no prediction found for sample number {idx}")
                meta["prediction"] = None

            actions= extract_action(meta["label"])
            if actions:
                action = actions[0] 
                tool_cfg_gold, _ = extract_tool_cfg(action)
            else:
                tool_cfg_gold = None
  
            if tool_cfg_gold: # supposed to call tool
                total_calls += 1
                if tool_cfg_pred:  # and tool was called
                    metrics["inst_acc"] += 1
                    if tool_cfg_pred["API_name"] not in TOOLS_LIST:
                        error_breakdown['name_not_in_tools'] += 1
                    if tool_cfg_gold["API_name"] == tool_cfg_pred["API_name"]: #correct tool was called
                        metrics["tool_acc"] += 1
                        if tool_cfg_gold["API_name"] == "Terminate":
                            ans_gold = tool_cfg_gold["API_params"].get('ans')
                            ans_pred = tool_cfg_pred["API_params"].get('ans',"")
                            question = meta["conversation"][0]["value"].split("\n\nQuestion:", 1)[1].strip()
                            #only for non gen samples
                            if not last_tool_in_gen(meta["conversation"]):
                                count_non_gen_samples +=1
                                metrics['answer_acc'] += sim_score(ans_gold,ans_pred,question)
                            else: 
                                count_gen_samples += 1  
                        if tool_cfg_gold["API_params"].keys() == tool_cfg_pred["API_params"].keys(): # correct arguments were passed
                            metrics["arg_acc1"] += 1
                            if tool_cfg_gold["API_params"] == tool_cfg_pred["API_params"]: # matching argument values
                                metrics["arg_acc2"] += 1
                            else:
                                error_breakdown['argument_values_not_matching'] += 1
                        else: 
                            error_breakdown['required_argument_missing'] += 1
                    else:
                        error_breakdown['incorrect_tool_called'] += 1
                else:
                    error_breakdown[error] += 1
             
        # normalize metrics            
        if total_calls > 0:
            for key in ["inst_acc", "tool_acc", "arg_acc1", "arg_acc2"]:
                metrics[key] /= total_calls
                metrics[key] *= 100
        if count_non_gen_samples > 0:
            metrics["answer_acc"] = (metrics["answer_acc"] /count_non_gen_samples) *100
        metrics["count_non_gen_samples"] = count_non_gen_samples
        metrics["count_gen_samples"] = count_gen_samples
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        logger.info(f"Metrics: {metrics}")
        logger.info(f"Error breakdown: {error_breakdown}")
        return {"metrics":metrics, "error_breakdown": error_breakdown}
    else:
        raise ValueError(f"Unsupported task type: '{task_type}'")
    