"""Regenerate LIBERO demonstration datasets for native mujoco 3.x physics.

The LIBERO demos (yifengzhu-hf/LIBERO-datasets, made with scripts/create_dataset.py) were recorded and
rendered with mujoco 2.3.x. For every demo this script:

1. rebuilds the env from the demo's model_file (as create_dataset.py does, so fixture placement matches
   the recording; asset paths mapped by postprocess_model_xml) and sets the demo's first state;
2. settles that state under the installed mujoco with libero.libero.utils.settle_utils.settle_current
   (penetrating objects lifted, floating objects lowered onto their support, wait action until at
   rest), then puts the robot joints back to the recorded start (the wait moves the arm by up to
   0.07 rad, and the recorded actions were made from the recorded pose); demos whose start is unstable
   are dropped. The tilt test of the init states is not applied: the recorded starts already contain
   objects that came to rest tilted under mujoco 2.3.x (the bowl on the ramekin is tilted 23.6 degrees
   in all 50 libero_spatial demos), and step 5 checks the result;
3. reloads the settled state the way a user replays a demo (env.reset(), reset_from_xml_string(model_file),
   sim.reset(), sim.set_state_from_flattened(states[0]), sim.forward()), so controller and solver warm
   start state do not carry over from settling and a replay of the new file follows the same path;
4. replays the recorded actions open loop and records the same fields as create_dataset.py:
   states[t] is the sim state before action t, obs[t] the observation after it;
5. keeps the demo if the task is successful after the last action.

Output: <dst>/<suite>/<task>_demo.hdf5 with the original layout (data/demo_k/{actions, states,
robot_states, rewards, dones, obs/{agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, ee_states,
gripper_states, joint_states}}, attrs init_state/model_file/num_samples), plus <task>_demo.report.json.
Each demo also has the attrs source_demo and fixture_poses (ControlEnv.get_fixture_poses() of the model_file
scene): a demo can be replayed exactly through model_file as in step 3, or in an env built from the bddl file
with env.reset(); env.set_init_state(np.concatenate([states[0], fixture_poses])).

Usage:
  python scripts/regenerate_demos_mj314.py --src <LIBERO-datasets> --dst <out> --suites libero_spatial
"""

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

import libero.libero  # noqa: E402  (first import may ask for the LIBERO config, keep it visible)
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.utils.settle_utils import (  # noqa: E402
    SettleParams,
    free_object_bodies,
    max_penetration,
    settle_current,
)

with contextlib.redirect_stdout(io.StringIO()):
    import robosuite.utils.transform_utils as T  # noqa: E402
    from libero.libero.envs import mujoco_compat  # noqa: E402
    from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402
    from libero.libero.utils.utils import postprocess_model_xml  # noqa: E402


def load_model_xml(xml_str):
    """postprocess_model_xml (robosuite and LIBERO asset paths mapped to this installation), and check
    that every referenced asset exists."""
    xml_str = postprocess_model_xml(xml_str, {})
    asset = ET.fromstring(xml_str).find("asset")
    files = [e.get("file") for e in asset.findall("mesh") + asset.findall("texture") if e.get("file")]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} assets referenced by the demo model are missing, e.g. {missing[:3]}")
    return xml_str


def make_env(bddl, env_kwargs):
    kw = dict(
        bddl_file_name=bddl,
        camera_names=env_kwargs["camera_names"],
        camera_heights=env_kwargs["camera_heights"],
        camera_widths=env_kwargs["camera_widths"],
        camera_depths=env_kwargs["camera_depths"],
        control_freq=env_kwargs["control_freq"],
        has_offscreen_renderer=True,
        use_camera_obs=True,
        ignore_done=True,
        reward_shaping=env_kwargs.get("reward_shaping", True),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        env = ControlEnv(**kw)
    return env


def controller_matches(env, recorded):
    live = env.env.robots[0].controller_config
    keys = ("type", "kp", "damping_ratio", "output_max", "output_min", "input_max", "input_min", "control_delta",
            "uncouple_pos_ori", "impedance_mode")
    return all(np.allclose(np.atleast_1d(live[k]), np.atleast_1d(recorded[k])) if not isinstance(live[k], str)
               else live[k] == recorded[k] for k in keys)


def load_state(env, model_xml, state):
    """Load a demo start the way scripts/create_dataset.py and users replaying a demo do."""
    env.reset()
    env.env.reset_from_xml_string(model_xml)
    env.env.sim.reset()
    env.env.sim.set_state_from_flattened(state)
    env.env.sim.forward()


def robot_qpos_indexes(env):
    robot = env.env.robots[0]
    return list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)


def replay(env, model_xml, start_state, actions, params):
    load_state(env, model_xml, start_state)
    sim = env.env.sim
    idx = robot_qpos_indexes(env)
    robot_qpos = sim.data.qpos[idx].copy()
    settle = settle_current(env, params)
    settle["robot_joint_shift"] = round(float(np.abs(sim.data.qpos[idx] - robot_qpos).max()), 6)
    sim.data.qpos[idx] = robot_qpos
    sim.data.qvel[:] = 0.0
    sim.forward()
    model, data = sim.model._model, sim.data._data
    settle["max_pen_robot_restored"] = round(max_penetration(model, data, free_object_bodies(model)), 6)
    settle["stable"] = settle["stable"] and settle["max_pen_robot_restored"] <= params.max_pen
    if not settle["stable"]:
        return None, settle
    settled = sim.get_state().flatten()
    model_xml = env.env.sim.model.get_xml()
    load_state(env, model_xml, settled)
    fixture_poses = env.get_fixture_poses()
    rec = {k: [] for k in ("states", "agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states",
                           "ee_states", "robot_states")}
    first_success = None
    for t, action in enumerate(actions):
        rec["states"].append(env.env.sim.get_state().flatten())
        obs, _, _, _ = env.step(action)
        rec["agentview_rgb"].append(obs["agentview_image"])
        rec["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"])
        rec["joint_states"].append(obs["robot0_joint_pos"])
        rec["gripper_states"].append(obs["robot0_gripper_qpos"])
        rec["ee_states"].append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))
        rec["robot_states"].append(env.env.get_robot_state_vector(obs))
        if first_success is None and env.env._check_success():
            first_success = t
    rec["success"] = bool(env.env._check_success())
    rec["first_success"] = first_success
    rec["model_xml"] = model_xml
    rec["fixture_poses"] = fixture_poses
    return rec, settle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with the original <suite>/<task>_demo.hdf5 files")
    ap.add_argument("--dst", required=True, help="output folder")
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--tasks", nargs="+", default=None, help="task names (file stem without _demo); default all")
    ap.add_argument("--demos", type=int, default=None, help="only the first N demos per task (for tests)")
    ap.add_argument("--exclude", nargs="*", default=[], help="task names to skip")
    args = ap.parse_args()
    params = SettleParams(max_tilt=180.0)  # no tilt test for recorded starts, see step 2

    status = mujoco_compat.status()
    if status["installed"]:
        sys.exit("the 2.3.7 compat patch is active; unset LIBERO_MUJOCO_COMPAT to regenerate under native physics")
    print("mujoco", mujoco.__version__, "| compat:", status["mode"], "| dst:", args.dst, flush=True)

    for suite in args.suites:
        files = sorted(glob.glob(os.path.join(args.src, suite, "*_demo.hdf5")))
        for src_path in files:
            name = os.path.basename(src_path)[: -len("_demo.hdf5")]
            if args.tasks and name not in args.tasks:
                continue
            out_dir = os.path.join(args.dst, suite)
            os.makedirs(out_dir, exist_ok=True)
            dst_path = os.path.join(out_dir, f"{name}_demo.hdf5")
            if name in args.exclude:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                with open(dst_path[: -len(".hdf5")] + ".report.json", "w") as f:
                    json.dump({"suite": suite, "task": name, "excluded": True, "reason": "--exclude"}, f)
                print(f"{suite}/{name}: EXCLUDED", flush=True)
                continue
            t0 = time.time()
            tmp_path = dst_path + f".{os.getpid()}.tmp"
            try:
                kept, report = regenerate_task(args, params, suite, name, src_path, tmp_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            os.replace(tmp_path, dst_path)
            with open(dst_path[: -len(".hdf5")] + ".report.json", "w") as f:
                json.dump({"suite": suite, "task": name, "mujoco": mujoco.__version__, "kept": kept,
                           "n_source": len(report), "demos": report}, f)
            reasons = {}
            for e in report:
                if not e["kept"]:
                    reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
            print(f"{suite}/{name}: kept {kept}/{len(report)} {reasons} {time.time() - t0:.0f}s", flush=True)


def regenerate_task(args, params, suite, name, src_path, tmp_path):
    """Regenerate the demos of one task file into tmp_path. Returns (kept, per-demo report)."""
    bddl = os.path.join(get_libero_path("bddl_files"), suite, f"{name}.bddl")
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as out:
        env_args = json.loads(src["data"].attrs["env_args"])
        env = make_env(bddl, env_args["env_kwargs"])
        try:
            if not controller_matches(env, env_args["env_kwargs"]["controller_configs"]):
                sys.exit(f"{src_path}: controller config differs from the recorded one")
            demos = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if args.demos:
                demos = demos[: args.demos]
            grp = out.create_group("data")
            for key in ("env_name", "env_args", "bddl_file_name", "macros_image_convention", "problem_info", "tag"):
                if key in src["data"].attrs:
                    grp.attrs[key] = src["data"].attrs[key]
            grp.attrs["regenerated_with"] = json.dumps({"mujoco": mujoco.__version__,
                                                        "source": os.path.relpath(src_path, args.src),
                                                        "settle": vars(params)})
            report, kept, total = [], 0, 0
            for ep in demos:
                g = src["data"][ep]
                actions = np.array(g["actions"][()])
                entry = {"demo": ep}
                try:
                    model_xml = load_model_xml(g.attrs["model_file"])
                    rec, settle = replay(env, model_xml, np.array(g["states"][0]), actions, params)
                except (FileNotFoundError, ValueError) as e:  # broken source demo (missing asset, bad xml)
                    entry["kept"], entry["reason"] = False, f"error: {type(e).__name__}: {e}"
                    report.append(entry)
                    continue
                entry["settle"] = {k: settle[k] for k in ("stable", "max_dxy", "max_pen_after", "max_lift", "max_drop",
                                                          "max_tilt_deg", "robot_joint_shift",
                                                          "max_pen_robot_restored", "steps")}
                if rec is None:
                    entry["kept"], entry["reason"] = False, "unstable start"
                elif not rec["success"]:
                    entry["kept"], entry["reason"] = False, "replay did not succeed"
                    entry["first_success"] = rec["first_success"]
                else:
                    write_demo(grp.create_group(f"demo_{kept}"), rec, actions, ep)
                    entry["kept"], entry["first_success"], entry["new_key"] = True, rec["first_success"], f"demo_{kept}"
                    kept += 1
                    total += len(actions)
                report.append(entry)
            grp.attrs["num_demos"] = kept
            grp.attrs["total"] = total
        finally:
            env.close()
    return kept, report


def write_demo(d, rec, actions, source_demo):
    o = d.create_group("obs")
    o.create_dataset("agentview_rgb", data=np.stack(rec["agentview_rgb"]))
    o.create_dataset("eye_in_hand_rgb", data=np.stack(rec["eye_in_hand_rgb"]))
    ee = np.stack(rec["ee_states"])
    o.create_dataset("ee_states", data=ee)
    o.create_dataset("ee_pos", data=ee[:, :3])
    o.create_dataset("ee_ori", data=ee[:, 3:])
    o.create_dataset("gripper_states", data=np.stack(rec["gripper_states"]))
    o.create_dataset("joint_states", data=np.stack(rec["joint_states"]))
    states = np.stack(rec["states"])
    d.create_dataset("actions", data=actions)
    d.create_dataset("states", data=states)
    d.create_dataset("robot_states", data=np.stack(rec["robot_states"]))
    rewards = np.zeros(len(actions), dtype=np.uint8)
    rewards[-1] = 1
    d.create_dataset("rewards", data=rewards)
    d.create_dataset("dones", data=rewards.copy())
    d.attrs["num_samples"] = len(actions)
    d.attrs["model_file"] = rec["model_xml"]
    d.attrs["init_state"] = states[0]
    d.attrs["source_demo"] = source_demo
    d.attrs["fixture_poses"] = rec["fixture_poses"]


if __name__ == "__main__":
    main()
