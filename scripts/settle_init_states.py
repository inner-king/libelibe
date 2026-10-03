"""Settle LIBERO init states under the installed mujoco (native physics).

The original init states were written right after placement sampling, without settling, and were
used with mujoco 2.3.x, whose box-box collider reported half the true penetration depth. Objects
therefore start up to ~3 cm inside the table or inside each other, or 6-16 cm above their support.
Under mujoco >= 3.4 the penetrations launch or move objects before the policy acts.

For every init state this script keeps the object layout and makes it physically consistent:
1. env.reset(), env.set_init_state(original state);
2. fixture pose: env.reset() places the fixtures (stove, cabinets, wine rack) at a random pose within
   a 2 cm region, and the original states do not record the pose they were sampled with. The script
   draws --fixture-candidates poses from the same samplers and keeps the one with the least
   penetration between objects and fixtures;
3. libero.libero.utils.settle_utils.settle_current: lift penetrating objects, lower floating objects
   onto their support, zero velocities, run the standard wait action until objects are at rest;
4. save [time, qpos, qvel] followed by the fixture poses (ControlEnv.get_fixture_poses()).
   ControlEnv.set_init_state() restores the fixture poses, so every reset reproduces the settled scene.

An init is unstable if it does not come to rest within --max-steps, the lift does not converge within
--max-lift, any object moves more than --max-dxy in xy while settling, a contact deeper than --max-pen
remains afterwards, or an object that touches another movable object ends tilted by more than
--max-tilt. An unstable init is replaced by a new placement from the LIBERO sampler (env.reset(), with
the fixture pose of that reset), settled the same way, up to --resample-attempts times. If no stable
placement is found for some init, the task gets a <file>.excluded marker with the reason instead of an
init file (benchmark.get_task_init_states then raises InitStatesUnavailable under native physics).

Output: <out-root>/<problem_folder>/<init_states_file> (an array of shape (n_inits, 1+nq+nv+7*n_fixtures)),
plus <init_states_file>.report.json with per-init settle reports.

Usage:
  LIBERO_INIT_STATES=original python scripts/settle_init_states.py --suites libero_spatial_lan --tasks 5
  LIBERO_INIT_STATES=original python scripts/settle_init_states.py --suites libero_10_lan --inits 5 \
      --out-root /tmp/settled
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import torch  # noqa: E402

import libero.libero  # noqa: E402  (first import may ask for the LIBERO config, keep it visible)
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.utils.settle_utils import SettleParams, free_object_bodies, settle_current  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from libero.libero import benchmark  # noqa: E402
    from libero.libero.envs import mujoco_compat  # noqa: E402
    from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

PRO_SUITES = [f"libero_{s}_{p}" for s in ("spatial", "object", "goal", "10") for p in ("lan", "object", "swap", "task")]


def source_sha256(bench, i):
    """sha256 of the original init file the states were settled from."""
    task = bench.get_task(i)
    with open(os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fixture_samplers(env):
    fixtures = env.env.fixtures_dict
    return [s for s in env.env.placement_initializer.samplers.values()
            if s.mujoco_objects and all(o.name in fixtures for o in s.mujoco_objects)]


def object_penetration(env):
    """Sum of penetration depths of all contacts that involve a movable object (m)."""
    model, data = env.env.sim.model._model, env.env.sim.data._data
    mujoco.mj_forward(model, data)
    free = free_object_bodies(model)
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.body_rootid[model.geom_bodyid[c.geom1]])
        b2 = int(model.body_rootid[model.geom_bodyid[c.geom2]])
        if b1 != b2 and (b1 in free or b2 in free):
            total += max(0.0, -float(c.dist))
    return total


def choose_fixture_pose(env, n_candidates):
    """Try the current fixture pose and n_candidates - 1 poses drawn from the fixture samplers; keep the
    one with the least object penetration. Returns (chosen index, penetration of candidate 0, chosen)."""
    names = sorted(env.env.fixtures_dict)
    current = env.get_fixture_poses()
    candidates = [current]
    samplers = fixture_samplers(env)
    for _ in range(n_candidates - 1 if samplers else 0):
        poses = current.copy()
        for sampler in samplers:
            for name, (pos, quat, _) in sampler.sample().items():
                k = names.index(name)
                poses[7 * k : 7 * k + 3] = pos
                poses[7 * k + 3 : 7 * k + 7] = quat
        candidates.append(poses)
    scores = []
    for poses in candidates:
        env.set_fixture_poses(poses)
        scores.append(object_penetration(env))
    best = int(np.argmin(scores))
    env.set_fixture_poses(candidates[best])
    mujoco.mj_forward(env.env.sim.model._model, env.env.sim.data._data)
    return best, scores[0], scores[best]


def settle_one(env, state, params, n_candidates):
    """Settle `state`, or a fresh sampler placement if `state` is None. Returns (state, report)."""
    env.reset()  # robosuite hard reset builds a new MjSim and places the fixtures
    fixture = None
    if state is not None:
        env.set_init_state(state)
        if n_candidates > 1:
            best, pen_reset, pen_best = choose_fixture_pose(env, n_candidates)
            fixture = {"candidate": best, "penetration_reset": round(pen_reset, 6),
                       "penetration_chosen": round(pen_best, 6)}
    report = settle_current(env, params)
    report["fixture_pose"] = fixture
    out = np.concatenate([env.get_sim_state(), env.get_fixture_poses()]).astype(np.float64)
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=PRO_SUITES)
    ap.add_argument("--tasks", nargs="+", type=int, default=None, help="task indices (default: all)")
    ap.add_argument("--inits", type=int, default=None, help="only the first N init states (for tests)")
    ap.add_argument("--out-root", default=None,
                    help="default: libero/libero/init_files_mj314 (required with --inits, so the shipped "
                         "files are not replaced by truncated ones)")
    defaults = SettleParams()
    ap.add_argument("--pen-tol", type=float, default=defaults.pen_tol, help="m; deeper supporting contacts are lifted")
    ap.add_argument("--speed-tol", type=float, default=defaults.speed_tol, help="m/s; settle stop criterion")
    ap.add_argument("--quiet-steps", type=int, default=defaults.quiet_steps, help="steps below --speed-tol to stop")
    ap.add_argument("--min-steps", type=int, default=defaults.min_steps)
    ap.add_argument("--max-steps", type=int, default=defaults.max_steps)
    ap.add_argument("--max-dxy", type=float, default=defaults.max_dxy, help="m; larger xy motion while settling = unstable")
    ap.add_argument("--max-pen", type=float, default=defaults.max_pen, help="m; deeper remaining contact = unstable")
    ap.add_argument("--max-lift", type=float, default=defaults.max_lift, help="m; larger lift = unstable")
    ap.add_argument("--max-tilt", type=float, default=defaults.max_tilt,
                    help="deg; larger tilt of an object touching another movable object = unstable")
    ap.add_argument("--max-drop", type=float, default=defaults.max_drop, help="m; support search depth")
    ap.add_argument("--fixture-candidates", type=int, default=16, help="fixture poses tried per original init")
    ap.add_argument("--resample-attempts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out_root is None:
        if args.inits:
            sys.exit("--inits writes truncated init files; give --out-root")
        args.out_root = os.path.join(get_libero_path("benchmark_root"), "init_files_mj314")
    params = SettleParams(pen_tol=args.pen_tol, speed_tol=args.speed_tol, quiet_steps=args.quiet_steps,
                          min_steps=args.min_steps, max_steps=args.max_steps, max_dxy=args.max_dxy,
                          max_pen=args.max_pen, max_lift=args.max_lift, max_tilt=args.max_tilt,
                          max_drop=args.max_drop)

    status = mujoco_compat.status()
    if status["installed"]:
        sys.exit("the 2.3.7 compat patch is active; unset LIBERO_MUJOCO_COMPAT to settle under native physics")
    if os.environ.get("LIBERO_INIT_STATES", "") != "original":
        sys.exit("set LIBERO_INIT_STATES=original so the original init files are read")
    print("mujoco", mujoco.__version__, "| compat:", status["mode"], "| out:", args.out_root, flush=True)

    bench_dict = benchmark.get_benchmark_dict()
    for suite in args.suites:
        bench = bench_dict[suite]()
        task_ids = args.tasks if args.tasks is not None else range(bench.n_tasks)
        for i in task_ids:
            t0 = time.time()
            task = bench.get_task(i)
            states = bench.get_task_init_states(i)
            if args.inits:
                states = states[: args.inits]
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            with contextlib.redirect_stdout(io.StringIO()):
                env = ControlEnv(bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False)
            env.seed(args.seed)
            out_states, reports, failed = [], [], []
            for k, s in enumerate(states):
                st, rep = settle_one(env, s, params, args.fixture_candidates)
                rep["init"], rep["source"] = k, "original"
                attempt = 0
                while not rep["stable"] and attempt < args.resample_attempts:
                    np.random.seed((args.seed * 1_000_003 + i * 10_007 + k * 101 + attempt) % 2**32)
                    st2, rep2 = settle_one(env, None, params, args.fixture_candidates)
                    rep2["init"], rep2["source"] = k, f"resampled:{attempt}"
                    rep2["original_report"] = rep if attempt == 0 else rep.get("original_report", rep)
                    st, rep = st2, rep2
                    attempt += 1
                if not rep["stable"]:
                    failed.append(k)
                out_states.append(st)
                reports.append(rep)
            fixture_names = sorted(env.env.fixtures_dict)
            env.close()
            out_dir = os.path.join(args.out_root, task.problem_folder)
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.join(out_dir, task.init_states_file)
            meta = {"suite": suite, "task": i, "name": task.name, "mujoco": mujoco.__version__,
                    "state_layout": "time, qpos, qvel, then pos+quat of each fixture root body sorted by name",
                    "fixtures": fixture_names, "source_sha256": source_sha256(bench, i),
                    "args": {k: v for k, v in vars(args).items() if k != "out_root"}, "settle": vars(params),
                    "failed_inits": failed, "inits": reports}
            with open(base + ".report.json", "w") as f:
                json.dump(meta, f)
            n_res = sum(r["source"] != "original" for r in reports)
            if failed:
                if os.path.exists(base):
                    os.remove(base)
                with open(base + ".excluded", "w") as f:
                    f.write(f"no stable placement under mujoco {mujoco.__version__} native physics for "
                            f"{len(failed)}/{len(states)} init states (original layout and "
                            f"{args.resample_attempts} resampled placements each).")
                status = f"EXCLUDED ({len(failed)} inits without a stable placement)"
            else:
                if os.path.exists(base + ".excluded"):
                    os.remove(base + ".excluded")
                torch.save(np.stack(out_states), base)
                status = f"saved, {n_res} resampled"
            print(f"{suite}[{i}] {len(out_states)} inits: {status}, max dxy {max(r['max_dxy'] for r in reports):.4f} m, "
                  f"max pen {max(r['max_pen_after'] for r in reports):.5f} m, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
