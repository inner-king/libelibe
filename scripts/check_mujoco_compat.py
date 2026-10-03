"""Check the mujoco 3.x setup of this fork.

Default (native mujoco physics, LIBERO_MUJOCO_COMPAT unset):
1. LIBERO-PRO suites load the init states settled for mujoco 3.x (libero/libero/init_files_mj314).
2. With those init states, reset with several seeds (each reset draws a new fixture pose), the standard
   wait (set_init_state + 10 steps of [0]*6+[-1]) moves no object by more than 1 mm, and the fixture
   poses stored in the init state are restored. Tasks: libero_10_lan 0 (objects used to launch 0.57 m),
   libero_spatial_lan 0, libero_spatial_lan 5 (bowl on the ramekin), and three tasks with objects
   against a fixture: libero_goal_swap 7 (plate on the stove edge), libero_10_swap 2 (frying pan on the
   stove), libero_spatial_swap 3 (bowl on the cookie box next to the cabinet).

With LIBERO_MUJOCO_COMPAT=1 (mujoco 2.3.7 compat patch, original init states), compare with values
measured on mujoco 2.3.7 + robosuite 1.4.0:
1. Two unit boxes, the second at z=1.5: box-box contact depth. 2.3.7: -0.25 (3.4+ native: -0.5).
2. libero_spatial task 5: bowl-ramekin xy distance after the wait. 2.3.7: median 0.0145 m, 0 over 5 cm.
3. libero_10 task 0: largest object rise during the wait. 2.3.7: median 0.0292 m, 0 over 5 cm.

Usage: MUJOCO_GL=egl python scripts/check_mujoco_compat.py [--inits N] [--seeds 0 1]
Exit code 0 if all checks pass, 1 otherwise.
"""

import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

# The first import of libero.libero asks about the dataset folder (input()) when ~/.libero/config.yaml
# (or $LIBERO_CONFIG_PATH/config.yaml) does not exist yet, so it must not run inside redirect_stdout.
import libero.libero  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from libero.libero import benchmark  # noqa: E402
    from libero.libero.envs import OffScreenRenderEnv, mujoco_compat  # noqa: E402

WAIT = [0.0] * 6 + [-1.0]
PRO_SUITES = [f"libero_{s}_{p}" for s in ("spatial", "object", "goal", "10") for p in ("lan", "object", "swap", "task")]


def check_config():
    here = os.path.realpath(os.path.dirname(libero.libero.__file__))
    root = os.path.realpath(get_libero_path("benchmark_root"))
    if root != here:
        print(f"[WARN] {libero.libero.config_file} points at {root}, not at this checkout ({here}); "
              f"bddl/init files are read from there. Set LIBERO_CONFIG_PATH to a new directory or run "
              f"set_libero_default_path() to use this checkout.")


def box_depth():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'
        '<body><freejoint/><geom type="box" size="1 1 1"/></body>'
        '<body pos="0 0 1.5"><freejoint/><geom type="box" size="1 1 1"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return [float(data.contact[i].dist) for i in range(data.ncon)]


def settle(suite, task_id, n_inits, measure, seeds=(0,)):
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    init_states = bench.get_task_init_states(task_id)[:n_inits]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    with contextlib.redirect_stdout(io.StringIO()):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
    names = list(env.env.objects_dict)
    values, fixture_err = [], 0.0
    n = None
    for seed in seeds:
        env.seed(seed)
        for state in init_states:
            env.reset()
            env.set_init_state(state)
            inner = env.env  # reset builds a new sim; read it after the reset
            n = 1 + inner.sim.model.nq + inner.sim.model.nv
            if len(state) > n:
                fixture_err = max(fixture_err, float(np.abs(env.get_fixture_poses() - state[n:]).max()))
            before = {k: inner.sim.data.body_xpos[inner.obj_body_id[k]].copy() for k in names}
            for _ in range(10):
                env.step(WAIT)
            after = {k: inner.sim.data.body_xpos[inner.obj_body_id[k]].copy() for k in names}
            values.append(measure(inner, names, before, after))
    has_fixtures = n is not None and len(init_states[0]) > n
    env.close()
    return np.array(values), (fixture_err if has_fixtures else None)


def ramekin_offset(inner, names, before, after):
    ramekin = next(n for n in names if "ramekin" in n)
    return float(np.linalg.norm(after[inner.obj_of_interest[0]][:2] - after[ramekin][:2]))


def max_rise(inner, names, before, after):
    return float(max(after[n][2] - before[n][2] for n in names))


def max_motion(inner, names, before, after):
    return float(max(np.linalg.norm(after[n] - before[n]) for n in names))


def report(passed, text):
    print(f"[{'PASS' if passed else 'FAIL'}] {text}")
    return passed


NATIVE_TASKS = (("libero_10_lan", 0), ("libero_spatial_lan", 0), ("libero_spatial_lan", 5), ("libero_goal_swap", 7),
                ("libero_10_swap", 2), ("libero_spatial_swap", 3))


def check_native(n_inits, seeds):
    ok = True
    source = benchmark._init_states_source()
    ok &= report(source == "mj314", f"init state source for LIBERO-PRO suites: {source} (expected mj314)")
    root = os.path.join(get_libero_path("benchmark_root"), "init_files_mj314")
    bench_dict = benchmark.get_benchmark_dict()
    missing, stale = [], []
    for suite in PRO_SUITES:
        with contextlib.redirect_stdout(io.StringIO()):
            bench = bench_dict[suite]()
        for i in range(bench.n_tasks):
            task = bench.get_task(i)
            path = os.path.join(root, task.problem_folder, task.init_states_file)
            if not (os.path.exists(path) or os.path.exists(path + ".excluded")):
                missing.append(f"{suite} {i}")
                continue
            source = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
            with open(path + ".report.json") as f:
                recorded = json.load(f).get("source_sha256")
            if os.path.exists(source):
                with open(source, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() != recorded:
                        stale.append(f"{suite} {i}")
    ok &= report(not missing, f"init_files_mj314 has a file or an .excluded marker for every task of the 16 LIBERO-PRO "
                 f"suites ({len(missing)} missing{': ' + ', '.join(missing[:5]) if missing else ''})")
    ok &= report(not stale, f"init_files_mj314 were settled from the current original init files ({len(stale)} tasks "
                 f"whose original init file changed{': ' + ', '.join(stale[:5]) if stale else ''}; rerun "
                 f"scripts/settle_init_states.py for them)")
    for suite, task_id in NATIVE_TASKS:
        try:
            motion, fixture_err = settle(suite, task_id, n_inits, max_motion, seeds)
        except (FileNotFoundError, benchmark.InitStatesUnavailable) as e:
            ok &= report(False, f"{suite} {task_id}: {e}")
            continue
        fixtures = "no fixtures" if fixture_err is None else f"fixture pose error after set_init_state {fixture_err:.1e}"
        passed = motion.max() <= 1e-3 and (fixture_err is None or fixture_err == 0.0)
        ok &= report(passed, f"{suite} {task_id}: largest object motion during the wait {motion.max():.2e} m over "
                     f"{len(motion)} resets (seeds {list(seeds)}; limit 1e-3 m), {fixtures}")
    excluded = sorted(glob.glob(os.path.join(get_libero_path("benchmark_root"), "init_files_mj314", "*", "*.excluded")))
    print(f"[INFO] excluded tasks under native physics: {len(excluded)}"
          + "".join(f"\n       {os.path.relpath(p, get_libero_path('benchmark_root'))}" for p in excluded))
    return ok


def check_compat(n_inits):
    ok = True
    depths = box_depth()
    ok &= report(len(depths) > 0 and all(abs(x + 0.25) < 1e-9 for x in depths),
                 f"box-box depth {depths}  (2.3.7: -0.25, 3.4+ native: -0.5)")
    offsets, _ = settle("libero_spatial", 5, n_inits, ramekin_offset)
    far = int((offsets > 0.05).sum())
    ok &= report(far == 0 and abs(np.median(offsets) - 0.0145) < 2e-3,
                 f"libero_spatial 5 bowl-ramekin: median {np.median(offsets):.4f} m, {far}/{len(offsets)} over 5 cm  "
                 f"(2.3.7: 0.0145 m, 0; 3.14 native: 0.0623 m, all)")
    rises, _ = settle("libero_10", 0, n_inits, max_rise)
    high = int((rises > 0.05).sum())
    ok &= report(high == 0 and abs(np.median(rises) - 0.0292) < 2e-3,
                 f"libero_10 0 max object rise: median {np.median(rises):.4f} m, {high}/{len(rises)} over 5 cm  "
                 f"(2.3.7: 0.0292 m, 0; 3.14 native: 0.5737 m)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inits", type=int, default=20, help="init states per task (max 50)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="env seeds (native mode)")
    args = parser.parse_args()

    status = mujoco_compat.status()
    print("mujoco", mujoco.__version__, "| compat status:", status)
    check_config()
    if status["installed"]:
        ok = check_compat(args.inits)
        print("all checks match mujoco 2.3.7" if ok else "some checks differ from mujoco 2.3.7")
    else:
        ok = check_native(args.inits, args.seeds)
        print("native mujoco setup OK" if ok else "native mujoco setup has failures")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
