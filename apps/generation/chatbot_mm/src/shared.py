#####################################
#              Variables            #
#####################################
import time
from .path import MLM_LOCAL_MODELS, LORA_LOCAL_MODELS

# Model variables
model_type = None   # transformers / llama
model_name = None
generator = None
tokenizer = None    # tokenizer / processor (VL model)
is_seq2seq = False
lora_names = []
multimodals = []

# Generation variables
generation_last_time = time.time()
generation_lock = None
generation_config = {}
generation_fn = None
preparation_fn = None
postprocess_fn = None

# Misc.
stop_everything = False
multi_user = False
verbose = True


#######################################
#              Arguments              #
#######################################
from pathlib import Path
from .utils import load_config, get_device

root_dir = Path(__file__).resolve().parent  # ../src

model_args_path = root_dir / "config/model_arguments.yaml"
model_args = load_config(model_args_path, return_type="namespace")

model_args.cpu_memory = str(model_args.cpu_memory)
model_args.gpu_memory = [str(x) for x in model_args.gpu_memory]

model_args.disk_cache_dir = root_dir.parents[3] / "temp/chatmm"
if model_args.disk_cache_dir.exists() is False:
    model_args.disk_cache_dir.mkdir()

model_config_path = root_dir / "config/model_config.yaml"
model_config = load_config(model_config_path, return_type="ordereddict")

user_config_path = root_dir / "config/user_config.yaml"
user_config = load_config(user_config_path, return_type="ordereddict")

device = get_device()

