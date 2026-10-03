import os
import xml.etree.ElementTree as ET
import robosuite
from robosuite.utils.mjcf_utils import find_elements
import numpy as np
import json
import torch
import random
from pathlib import Path

DIR = os.path.dirname(__file__)


def libero_asset_path(old_path):
    """Map a LIBERO asset path recorded on another machine (.../chiliocosm/assets/<rest> or
    .../libero/assets/<rest>) to the same file in this installation's assets folder.
    Returns None if the path is not a LIBERO asset path or the file does not exist here."""
    parts = old_path.replace("\\", "/").split("/")
    idx = [k for k in range(1, len(parts)) if parts[k] == "assets" and parts[k - 1] in ("chiliocosm", "libero")]
    if not idx:
        return None
    from libero.libero import get_libero_path

    new_path = os.path.normpath(os.path.join(get_libero_path("assets"), *parts[idx[-1] + 1 :]))
    return new_path if os.path.exists(new_path) else None


def postprocess_model_xml(xml_str, cameras_dict={}):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found: robosuite asset paths are mapped to the
    installed robosuite, LIBERO asset paths (libero_asset_path) to this installation's assets.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" not in old_path_split:
            new_path = libero_asset_path(old_path)
            if new_path is not None:
                elem.set("file", new_path)
            continue
        ind = max(
            loc for loc, val in enumerate(old_path_split) if val == "robosuite"
        )  # last occurrence index
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")
    return ET.tostring(root, encoding="utf8").decode("utf8")


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.0


def reconstruct_image_output(img_array):
    # return (img_array + 1.) / 2. * 255.
    return img_array * 255.0


def update_env_kwargs(env_kwargs, **kwargs):
    for (k, v) in kwargs.items():
        env_kwargs[k] = v
