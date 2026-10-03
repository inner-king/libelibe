"""mujoco 2.3.7 contact and kinematics behaviour for LIBERO on mujoco 3.9-3.14.

LIBERO init states and demos were produced with mujoco 2.3.x. Four later mujoco changes alter
LIBERO scenes (objects launched off tables, a bowl sliding off a ramekin before the policy acts):

1. Box-box collider. 3.4.0 (commit 88383684) doubled the depth reported on the face path and
   3.12.0 rewrote the collider. All LIBERO object and fixture collision geoms are boxes.
2. Convex collision. nativeccd became the default in 3.3.0 and multiccd in 3.8.0. This changes
   gripper (mesh) to object (box) contacts.
3. Newton solver termination. Since 3.11.0 the solver stops earlier at the default tolerance 1e-8.
4. Quaternion normalization. 2.3.7 mj_kinematics first normalizes the qpos quaternions in place,
   unconditionally. 3.x normalizes a copy and skips |norm-1| <= 1e-15, which gives a different
   geom_xmat for init quaternions with |q|-1 = 2.2e-16 (spurious box-box contacts).

install() changes the current process as follows:

1. Writes compat_BoxBox_old (port of the 2.3.7 mjc_BoxBox, compat_237.c) into
   mjCOLLISIONFUNC[BOX][BOX] of the loaded libmujoco.
2-3. Wraps robosuite MjSim.__init__ so every new model gets
   opt.disableflags |= mjDSBL_NATIVECCD | mjDSBL_MULTICCD and opt.tolerance = 1e-12.
   robosuite builds a new MjSim on every hard reset, so this persists across env.reset().
4. Wraps mujoco.mj_step, mujoco.mj_forward and mujoco.mj_step1 (the three entry points robosuite
   and LIBERO call; each runs mj_kinematics) to apply compat_normalizeQuat_237, the 2.3.7
   in-place normalization, first. Only the first of the two normalizations inside 2.3.7
   mj_kinematics can be reproduced from outside; the second differs in the last bit.

The patches are process-global, so every process that steps a simulation needs them, and
every MjModel/MjData stepped through mujoco.mj_step/mj_forward/mj_step1 in the process gets its
qpos and mocap quaternions normalized in place.
libero.libero.envs calls install() on import, so SubprocVectorEnv workers that import LIBERO get
them as well. Code that binds the functions before install() (from mujoco import mj_step) or
calls other kinematics entry points (mj_kinematics, mj_fwdPosition, mj_forwardSkip) skips (4).

LIBERO_MUJOCO_COMPAT: unset, "", "1", "on", "true", "yes" = on; "0", "false", "off", "no" =
native mujoco physics; "force" = also install on mujoco versions newer than the allowed range
(3.9.0-3.14.x; only 3.14.0 was validated). Any other value warns and means on.
"""

import ctypes
import functools
import glob
import hashlib
import os
import re
import subprocess
import sys
import threading

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "compat_237.c")
_ENV_VAR = "LIBERO_MUJOCO_COMPAT"
_TOLERANCE = 1e-12
# -ffp-contract=off: FMA contraction changes rounding and breaks bit parity with the 2.3.7 wheel
_CFLAGS = ("-O2", "-ffp-contract=off", "-fPIC", "-shared")
# [min, max): versions with the mjPreContact interface and 8 box-box precontact slots; validated on 3.14.0
_TESTED = ((3, 9, 0), (3, 15, 0))
_WRAPPED = ("mj_step", "mj_forward", "mj_step1")
_MARK = "_libero_mujoco_compat"
_HINT = f"Set {_ENV_VAR}=0 to run with native mujoco physics."

_lock = threading.Lock()
_state = {"installed": False, "mode": None, "so": None, "lib": None, "fn": None}


def _version():
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", mujoco.__version__)
    return tuple(int(x) for x in m.groups()) if m else None


def _mode():
    value = os.environ.get(_ENV_VAR, "1").strip().lower()
    if value in ("0", "false", "off", "no"):
        return "off"
    if value == "force":
        return "force"
    if value not in ("", "1", "on", "true", "yes"):
        print(f"[libero] {_ENV_VAR}={value!r} is not recognized; treating it as on.", file=sys.stderr)
    return "on"


def _so_name():
    with open(_SRC, "rb") as f:
        key = f.read() + " ".join(_CFLAGS).encode() + mujoco.__version__.encode()
    return f"_compat_237-mujoco{mujoco.__version__}-{hashlib.sha256(key).hexdigest()[:12]}.so"


def _build_dirs():
    yield _HERE
    yield os.path.join(os.path.expanduser("~"), ".cache", "libero", "mujoco_compat")


def _find_so():
    for d in _build_dirs():
        path = os.path.join(d, _so_name())
        if os.path.exists(path):
            return path
    return None


def _shared_object():
    """Path of the compiled compat_237.c for the installed mujoco, building it if needed.

    The file name carries a hash of the source, the compiler flags and the mujoco version, so a
    build from another checkout, other flags or another mujoco is never reused.
    """
    path = _find_so()
    if path:
        return path
    name = _so_name()
    include = os.path.join(os.path.dirname(mujoco.__file__), "include")
    errors = []
    for d in _build_dirs():
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            errors.append(f"{d}: {e}")
            continue
        if not os.access(d, os.W_OK):
            errors.append(f"{d}: not writable")
            continue
        path = os.path.join(d, name)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        cmd = ["gcc", *_CFLAGS, f"-I{include}", "-o", tmp, _SRC, "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(f"gcc not found; it is needed to build {name}. {_HINT}") from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"building {name} failed:\n{' '.join(cmd)}\n{e.stderr}{_HINT}") from None
        os.replace(tmp, path)  # atomic, so concurrent workers never load a partial file
        return path
    raise RuntimeError(f"no writable directory to build {name}: {errors}. {_HINT}")


def _libmujoco():
    # RTLD_NOLOAD only succeeds if the object is already mapped by the mujoco bindings, so the
    # table patched below is the one mj_step uses.
    errors = []
    for path in sorted(glob.glob(os.path.join(os.path.dirname(mujoco.__file__), "libmujoco.so*"))):
        try:
            return ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError as e:
            errors.append(str(e))
    raise OSError(f"no loaded libmujoco.so* next to {mujoco.__file__}: {errors}")


def _boxbox_slot(lib):
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    ntypes = int(mujoco.mjtGeom.mjNGEOMTYPES)
    base = ctypes.addressof(ctypes.c_void_p.in_dll(lib, "mjCOLLISIONFUNC"))
    return ctypes.c_void_p.from_address(base + ctypes.sizeof(ctypes.c_void_p) * (box * ntypes + box))


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]


def _symbol_name(addr):
    info = _DlInfo()
    if ctypes.CDLL(None).dladdr(ctypes.c_void_p(addr), ctypes.byref(info)) and info.dli_sname:
        return info.dli_sname.decode()
    return None


def _apply_options(model):
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    model.opt.tolerance = _TOLERANCE


def _patch_mjsim():
    try:
        from robosuite.utils import binding_utils
    except ImportError as e:
        raise RuntimeError(f"importing robosuite failed: {e}. {_HINT}") from None

    sim_cls = binding_utils.MjSim
    if sim_cls.__dict__.get(_MARK):
        return  # already wrapped, e.g. by this module before an importlib.reload
    init = sim_cls.__init__

    @functools.wraps(init)
    def __init__(self, model, *args, **kwargs):
        _apply_options(model)
        init(self, model, *args, **kwargs)

    sim_cls.__init__ = __init__
    setattr(sim_cls, _MARK, True)


def _mujoco_wrapped():
    return all(getattr(getattr(mujoco, name), _MARK, False) for name in _WRAPPED)


def _patch_mujoco(normalize):
    for name in _WRAPPED:
        func = getattr(mujoco, name)
        if getattr(func, _MARK, False):
            continue  # already wrapped, e.g. by this module before an importlib.reload

        def wrapper(m, d, *args, _func=func, _name=name, **kwargs):
            if not (isinstance(m, mujoco.MjModel) and isinstance(d, mujoco.MjData)):
                return _func(m, d, *args, **kwargs)  # let the binding raise its TypeError
            if _name == "mj_step":
                nstep = kwargs.pop("nstep", args[0] if args else 1)
                for _ in range(nstep):  # 2.3.7 normalizes once per step
                    normalize(m._address, d._address)
                    _func(m, d)
                return None
            normalize(m._address, d._address)
            return _func(m, d, *args, **kwargs)

        functools.update_wrapper(wrapper, func)
        wrapper.__module__, wrapper.__qualname__ = "mujoco", name
        setattr(wrapper, _MARK, True)
        setattr(mujoco, name, wrapper)


def install():
    """Apply the compat patches once per process. Returns True if they are active."""
    with _lock:
        if _state["installed"]:
            _patch_mjsim()  # no-ops unless robosuite or mujoco wrappers were reset by a reload
            _patch_mujoco(_state["so"].compat_normalizeQuat_237)
            return True
        mode = _mode()
        if mode == "off":
            _state["mode"] = "off"
            return False
        version = _version()
        if version is not None and version < (3, 0, 0):
            return False  # mujoco 2.x is the reference behaviour
        if version is None:
            _state["mode"] = "out_of_range"
            print(f"[libero] could not parse the mujoco version {mujoco.__version__!r}; the 2.3.7 "
                  f"compat patch is not installed and physics is native.", file=sys.stderr)
            return False
        if version < _TESTED[0]:
            _state["mode"] = "out_of_range"
            print(f"[libero] mujoco {mujoco.__version__}: the 2.3.7 compat patch needs mujoco "
                  f">= 3.9.0 (mjPreContact interface); running with native physics, which "
                  f"differs from 2.3.7.", file=sys.stderr)
            return False
        if version >= _TESTED[1] and mode != "force":
            _state["mode"] = "out_of_range"
            print(f"[libero] mujoco {mujoco.__version__} is newer than the range the 2.3.7 compat "
                  f"patch allows (3.9.0-3.14.x, validated on 3.14.0); running with native physics. "
                  f"{_ENV_VAR}=force installs it anyway.", file=sys.stderr)
            return False

        try:
            lib = _libmujoco()
            so = ctypes.CDLL(_shared_object())
        except OSError as e:
            raise RuntimeError(f"loading the mujoco compat library failed: {e}. {_HINT}") from None
        slot = _boxbox_slot(lib)
        orig = ctypes.cast(lib.mjc_BoxBox, ctypes.c_void_p).value
        fn = ctypes.cast(so.compat_BoxBox_old, ctypes.c_void_p).value
        if slot.value not in (orig, fn) and _symbol_name(slot.value) != "compat_BoxBox_old":
            raise RuntimeError(f"mjCOLLISIONFUNC[BOX][BOX] is {slot.value:#x}, expected mjc_BoxBox "
                               f"{orig:#x}; it was patched by something else. {_HINT}")
        normalize = so.compat_normalizeQuat_237
        normalize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        normalize.restype = None
        _patch_mjsim()
        _patch_mujoco(normalize)
        slot.value = fn
        _state.update(installed=True, mode=mode, so=so, lib=lib, fn=fn)
        return True


def status():
    """Patch state read from the process (collider slot, wrappers), plus call counters."""
    out = {"mujoco": mujoco.__version__, "mode": _state["mode"]}
    try:
        slot = _boxbox_slot(_libmujoco()).value
        out["boxbox_slot_is_compat"] = _symbol_name(slot) == "compat_BoxBox_old"
    except OSError:
        out["boxbox_slot_is_compat"] = False
    binding_utils = sys.modules.get("robosuite.utils.binding_utils")
    out["mjsim_wrapped"] = bool(binding_utils and binding_utils.MjSim.__dict__.get(_MARK))
    out["mujoco_wrapped"] = _mujoco_wrapped()
    out["installed"] = out["boxbox_slot_is_compat"] and out["mjsim_wrapped"] and out["mujoco_wrapped"]
    so = _state["so"]
    if so is None and out["installed"] and _find_so():
        try:  # loaded by an earlier copy of this module (e.g. before importlib.reload)
            so = ctypes.CDLL(_find_so(), mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            so = None
    if so is not None:
        for k in ("boxbox_ncall", "boxbox_noverflow", "boxbox_nmax", "normquat_ncall"):
            out[k] = ctypes.c_long.in_dll(so, f"compat_{k}").value
    return out
