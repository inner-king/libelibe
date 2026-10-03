"""Check that libero.libero.envs.mujoco_compat reproduces mujoco 2.3.7 behaviour.

Runs three checks and compares them with values measured on mujoco 2.3.7 + robosuite 1.4.0:

1. Two unit boxes, the second at z=1.5: box-box contact depth. 2.3.7: -0.25 (3.4+ native: -0.5).
2. libero_spatial task 5 (bowl on the ramekin): after set_init_state and 10 steps of [0]*6+[-1],
   xy distance between the bowl and the ramekin. 2.3.7: median 0.0145 m, 0 of 50 over 5 cm
   (3.14 native: median 0.0623 m, 50 of 50 over 5 cm).
3. libero_10 task 0 (LIVING_ROOM_SCENE2): largest object rise over the same 10 steps.
   2.3.7: median 0.0292 m, 0 of 50 over 5 cm (3.14 native: median 0.5737 m).

Usage: MUJOCO_GL=egl python scripts/check_mujoco_compat.py [--inits N]
Exit code 0 if all checks match 2.3.7, 1 otherwise.
"""

import argparse
import contextlib
import io
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


def check_config():
    here = os.path.realpath(os.path.dirname(libero.libero.__file__))
    root = os.path.realpath(get_libero_path("benchmark_root"))
    if root != here:
        print(f"[WARN] {libero.libero.config_file} points at {root}, not at this checkout ({here}); "
              f"bddl/init files are read from there. Set LIBERO_CONFIG_PATH to a new directory or run "
              f"set_libero_default_path() to use this checkout.")

WAIT = [0.0] * 6 + [-1.0]


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


def settle(suite, task_id, n_inits, measure):
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    init_states = bench.get_task_init_states(task_id)[:n_inits]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    with contextlib.redirect_stdout(io.StringIO()):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
    inner = env.env
    names = list(inner.objects_dict)
    values = []
    for state in init_states:
        env.seed(0)
        env.reset()
        env.set_init_state(state)
        before = {n: inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names}
        for _ in range(10):
            env.step(WAIT)
        after = {n: inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in names}
        values.append(measure(inner, names, before, after))
    env.close()
    return np.array(values)


def ramekin_offset(inner, names, before, after):
    ramekin = next(n for n in names if "ramekin" in n)
    return float(np.linalg.norm(after[inner.obj_of_interest[0]][:2] - after[ramekin][:2]))


def max_rise(inner, names, before, after):
    return float(max(after[n][2] - before[n][2] for n in names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inits", type=int, default=50, help="init states per task (max 50)")
    args = parser.parse_args()

    print("mujoco", mujoco.__version__, "| compat status:", mujoco_compat.status())
    check_config()
    ok = True

    depths = box_depth()
    passed = len(depths) > 0 and all(abs(x + 0.25) < 1e-9 for x in depths)
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] box-box depth {depths}  (2.3.7: -0.25, 3.4+ native: -0.5)")

    offsets = settle("libero_spatial", 5, args.inits, ramekin_offset)
    far = int((offsets > 0.05).sum())
    passed = far == 0 and abs(np.median(offsets) - 0.0145) < 2e-3
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] libero_spatial 5 bowl-ramekin: median {np.median(offsets):.4f} m, "
          f"{far}/{len(offsets)} over 5 cm  (2.3.7: 0.0145 m, 0; 3.14 native: 0.0623 m, all)")

    rises = settle("libero_10", 0, args.inits, max_rise)
    high = int((rises > 0.05).sum())
    passed = high == 0 and abs(np.median(rises) - 0.0292) < 2e-3
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] libero_10 0 max object rise: median {np.median(rises):.4f} m, "
          f"{high}/{len(rises)} over 5 cm  (2.3.7: 0.0292 m, 0; 3.14 native: 0.5737 m)")

    print("all checks match mujoco 2.3.7" if ok else "some checks differ from mujoco 2.3.7")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
