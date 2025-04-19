from ..utils import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

config_loader_hflibs = load_config(str(CONFIG_DIR / "loader_HFlibs.yaml"))
config_loader_llamacpp = load_config(str(CONFIG_DIR / "loader_llamacpp.yaml"))
config_loader_exllama = load_config(str(CONFIG_DIR / "loader_ExLlama.yaml"))
config_loader_tensorrt = load_config(str(CONFIG_DIR / "loader_TensorRT_LLM.yaml"))
