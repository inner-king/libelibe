from .robosuite_compat import apply as _apply_robosuite_compat
from .mujoco_compat import install as _install_mujoco_compat

_apply_robosuite_compat()
_install_mujoco_compat()

from .bddl_base_domain import TASK_MAPPING
from .base_object import OBJECTS_DICT
from .problems import *
from .robots import *
from .arenas import *
from .env_wrapper import OffScreenRenderEnv, SegmentationRenderEnv
from .venv import SubprocVectorEnv, DummyVectorEnv
