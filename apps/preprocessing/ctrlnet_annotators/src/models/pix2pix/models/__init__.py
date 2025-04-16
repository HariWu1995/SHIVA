"""
This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""
import importlib
from pathlib import Path
from types import ModuleType

from .base_model import BaseModel


def import_module_from_file(filepath):
    # Get absolute path and module name
    abs_path = os.path.abspath(filepath)
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not import module from {filepath}")


def find_model_using_name(model_name):
    """
    Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will be instantiated. 
    It has to be a subclass of BaseModel, and it is case-insensitive.
    """
    try:
        # model_dir = "annotator.leres.pix2pix.models."
        model_dir = "apps.preprocessing.ctrlnet_annotators.src.models.pix2pix.models."
        model_filename = model_dir + model_name
        model_lib = importlib.import_module(model_filename)

    except ModuleNotFoundError:
        model_dir = Path(__file__).resolve().parents[0]
        model_filename = str(model_dir / f"{model_name}.py")
        model_lib = import_module_from_file(model_filename)
    
    target_model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, clss in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower() \
        and issubclass(clss, BaseModel):
            target_model = clss
    assert target_model is not None, \
        f"In {model_filename}.py, there should be a subclass of BaseModel "\
        f"with class name that matches {target_model_name} in lowercase."
        
    return target_model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """
    Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
