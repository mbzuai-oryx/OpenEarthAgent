import json
import yaml
import os
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import io
import base64
import json
import os
import threading
import fcntl

_file_locks = {}
_lock_lock = threading.Lock()

def _get_file_lock(filepath):
    """Get a lock for a specific file path."""
    with _lock_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]

def load_json_file(filepath):
    '''
        将json文件读取成为列表或词典，线程安全
    '''
    with _get_file_lock(filepath):
        with open(filepath, 'r', encoding="UTF-8") as file:
            data = json.load(file)
    return data

def write_json_file(data, filepath):
    '''
        线程安全地写入JSON文件
    '''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with _get_file_lock(filepath):
        with open(filepath, 'w', encoding="UTF-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def process_jsonl(file_path):
    '''
        将jsonl文件转换为装有dict的列表，线程安全
    '''
    data = []
    with _get_file_lock(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                data.append(json_obj)
    return data

def write_jsonl(data, file_path):
    '''
        将list[dict]写入jsonl文件，线程安全
    '''
    with _get_file_lock(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                file.write(line + '\n')

def merge_jsonl(input_file_dir, output_filepath):
    '''
        将源文件夹内的所有jsonl文件合并为一个jsonl文件，线程安全
    '''
    filepaths = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir)]
    merged_data = []
    
    # First read all files
    for filepath in filepaths:
        with _get_file_lock(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    merged_data.append(data)
    
    # Then write to output
    with _get_file_lock(output_filepath):
        with open(output_filepath, 'w', encoding='utf-8') as output_file:
            for data in merged_data:
                output_file.write(json.dumps(data, ensure_ascii=False) + '\n')

def append_jsonl(data, filename):
    '''
        追加数据到jsonl文件，线程安全
    '''
    with _get_file_lock(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
        
def load_txt_file_as_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [line.strip().replace("\n","") for line in data]
    return data

def load_txt_file_as_str(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def write_txt_file(data, filepath):
    for item in data:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(item + '\n')
            
            
def print_rank0(msg):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg)
    else:
        print(msg)

def str2list(input_str):
    if isinstance(input_str,str):
        raw_list = input_str.strip().replace("\n","").split(",")
        new_list = []
        for item in raw_list:
            new_list.append(item.strip())
        return new_list
    elif isinstance(input_str,list):
        return input_str
    else:
        raise TypeError("input_str should be str or list")

def get_two_words(word1,word2):
    if word1 < word2:
        return f"{word1},{word2}"
    else:
        return f"{word2},{word1}"
    
 
def load_yaml_file(filepath):
    with open(filepath, 'r',encoding="UTF-8") as file:
        data = yaml.safe_load(file)
    return data

def write_yaml_file(data, filepath):
    with open(filepath, 'w',encoding="UTF-8") as file:
        yaml.dump(data, file,indent=4)
        
def tqdm_rank0(total, desc):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            pbar = tqdm(total=total, desc=desc)
            return pbar
        else:
            return None
    else:
        pbar = tqdm(total=total, desc=desc)
        return pbar

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True
    
    
def gather_dict_lists(local_dict_list):
    '''
        使用all_gather_object收集所有进程的数据
    '''
    if dist.is_available() and dist.is_initialized():
        # 获取总进程数
        world_size = dist.get_world_size()

        # 准备接收对象的空列表，每个进程分配一个 None
        gathered_dict_lists = [None for _ in range(world_size)]

        # 收集所有进程的数据
        dist.all_gather_object(gathered_dict_lists, local_dict_list)

        # 合并所有进程的数据到一个完整的列表
        final_merged_list = [item for sublist in gathered_dict_lists for item in sublist]
        return final_merged_list
    else:
        return local_dict_list

def setup_proxy():
    AD_NAME="songmingyang"
    encrypted_password="dSpydxsxxhKix63HfIFhjwnZLEInXEDawSoMD35G1IT2CygKnHsJqG9ZHbEP"
    new_proxy_address=f"http://{AD_NAME}:{encrypted_password}@10.1.20.50:23128/"
    # 设置环境变量
    os.environ['http_proxy'] = new_proxy_address
    os.environ['https_proxy'] = new_proxy_address
    os.environ['HTTP_PROXY'] = new_proxy_address
    os.environ['HTTPS_PROXY'] = new_proxy_address
    

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))
    

def b64_encode(img):
    '''
        将图片转换为base64编码
    '''
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def pil_to_base64(img: Image.Image, url_format = False) -> str:
    """
    Convert a PIL image to a base64 encoded string.
    
    Args:
        img (Image.Image): The PIL image to convert.
        
    Returns:
        str: Base64 encoded string representation of the image.
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if url_format:
        img_str = f"data:image/jpeg;base64,{img_str}"
    return img_str

def base64_to_pil(b64_str: str) -> Image.Image:
    """
    Convert a base64 encoded string into a PIL image.
    
    Args:
        b64_str (str): The base64 encoded image string.
        
    Returns:
        Image.Image: The resulting PIL image.
    """
    # Remove the data URI scheme if present
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split("base64,")[-1]
    return load_image_from_base64(b64_str)


def url_pil_to_base64(image):
    base64_str = b64_encode(image)
    base64_str = "data:image/jpeg;base64," + base64_str
    return base64_str

def url_base64_to_pil(b64_str):
    return base64_to_pil(b64_str)


def load_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    else:
        assert isinstance(image, str)
        if os.path.exists(image):
            return Image.open(image).convert('RGB')
        else:
            return load_image_from_base64(image)

