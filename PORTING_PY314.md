# Running LIBERO-PRO on Python 3.14 / mujoco 3.14

This repository is [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) master `eafdb80` (2026-03-23)
plus the changes below.

1. LIBERO-PRO runs on Python 3.14, torch 2.14, mujoco 3.14 and unmodified robosuite 1.4.0.
2. By default the simulation uses native mujoco 3.14 physics. LIBERO init states and demos were made with mujoco
   2.3.x and are not consistent with it (objects start up to 3 cm inside the table or 16 cm above it and launch or
   slide before the policy acts). This fork ships LIBERO-PRO init states settled for mujoco 3.x
   (`libero/libero/init_files_mj314/`, with the fixture poses stored in each state) and a script that regenerates
   the LIBERO demos under mujoco 3.x. The four original LIBERO suites (`libero_spatial`, `libero_object`,
   `libero_goal`, `libero_10`) and `libero_90` have no settled init states and use the original ones (§4.1).
3. Optionally (`LIBERO_MUJOCO_COMPAT=1`), a runtime patch reproduces mujoco 2.3.7 contact physics with the original
   init states, for comparison with published LIBERO / LIBERO-PRO numbers.

---

## 1. Installation

```bash
git clone https://github.com/inner-king/libelibe.git && cd libelibe
uv venv --python 3.14.4 .venv
uv pip install --python .venv/bin/python -r requirements-py314.txt
uv pip install --python .venv/bin/python -e .
echo n | .venv/bin/python -c "import libero.libero"                  # creates the LIBERO path config (see below)
MUJOCO_GL=egl .venv/bin/python scripts/check_mujoco_compat.py    # all checks must PASS
```

On first import, if `~/.libero/config.yaml` (or `$LIBERO_CONFIG_PATH/config.yaml`) does not exist, LIBERO asks
through `input()` whether to set a custom dataset folder. Answering `n` writes a config that points at this
clone's `libero/libero/`. If a config written by another LIBERO checkout already exists, bddl/init files are read
from that checkout. In that case use a separate `LIBERO_CONFIG_PATH` per clone, or repoint it with
`.venv/bin/python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"`.
`scripts/check_mujoco_compat.py` prints `[WARN]` when the config points at another checkout.

| Item | Value |
|---|---|
| Python | 3.14.4 (uv) |
| Main packages | mujoco 3.14.0, robosuite 1.4.0 (unmodified, PyPI), torch 2.14.1 (PyPI, CUDA 13), numpy 1.26.4, gym 0.26.2; full list in `requirements-py314.txt` |
| Build tool | gcc, only for the optional compat mode (§5): `compat_237.c` is compiled when the patch is first installed |
| Rendering | `MUJOCO_GL=egl` |

The LIBERO-PRO bddl files are not in this repository. As in the upstream README, copy `bddl_files/*` from the HF
dataset [`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) into
`libero/libero/bddl_files/` (and `init_files/*` into `libero/libero/init_files/` for the original init states used
by the compat mode). The files of the four original LIBERO suites and libero_90 are in git.

## 2. Changes against upstream

| File | Change | Reason |
|---|---|---|
| `libero/libero/benchmark/__init__.py` | `torch.load(..., weights_only=False)`; `get_task_init_states` reads `init_files_mj314` under native mujoco >= 3.4 physics (§4.1) | torch >= 2.6 defaults to `weights_only=True`, which fails on the init files; init states consistent with the physics in use |
| `libero/libero/envs/env_wrapper.py` `ControlEnv.reset` | removed the 2-line `finally: continue` | Python 3.14 PEP 765 SyntaxWarning. Behaviour change: before, every exception (KeyboardInterrupt included) was swallowed and retried forever; now only `RandomizationError` is retried |
| `libero/libero/envs/env_wrapper.py` `ControlEnv` | new `get_fixture_poses()` / `set_fixture_poses()`; `set_state` restores the fixture poses when the state is longer than `1 + nq + nv` | fixture poses are not part of the sim state (§4.1); states of the original length behave as before |
| `libero/libero/utils/utils.py` `postprocess_model_xml` | also maps LIBERO asset paths recorded on another machine (`.../chiliocosm/assets/...`, `.../libero/assets/...`) to this installation's assets, when the file exists here | demo `model_file` XMLs hold absolute asset paths of the machine that wrote them (§4.2) |
| `libero/__init__.py` (new, empty) | added | Without it the editable-install finder maps no package and `import libero` fails outside the repo root |
| `libero/libero/envs/__init__.py`, first 5 lines | calls `robosuite_compat.apply()` and `mujoco_compat.install()` | §2.1, §5 |
| `libero/libero/envs/robosuite_compat.py` (new) | replaces three robosuite 1.4.0 methods at import | §2.1 |
| `libero/libero/init_files_mj314/` (new) | init states of the 16 LIBERO-PRO suites settled under mujoco 3.14, with fixture poses, and per-task reports | §4.1 |
| `libero/libero/utils/settle_utils.py`, `scripts/settle_init_states.py`, `scripts/regenerate_demos_mj314.py` (new) | settling and regeneration tools | §4 |
| `libero/libero/envs/mujoco_compat/` (new) | `__init__.py`, `compat_237.c`, `LICENSE`, `.gitignore`: optional mujoco 2.3.7 compat patch | §5. `compat_237.c` is derived from mujoco 2.3.7 and is Apache-2.0; `LICENSE` is that license text (the rest of the repo is MIT). The build output `_compat_237-*.so` is git-ignored |
| `scripts/check_mujoco_compat.py` (new) | install check for the native setup or the compat mode | §1 |
| `requirements-py314.txt` (new) | `uv pip freeze` of the validated environment | §1 |

### 2.1 robosuite 1.4.0 on mujoco 3.x (`robosuite_compat.py`)

robosuite 1.4.0 from PyPI, unmodified, fails on mujoco 3.14 in two code paths that LIBERO uses:

| robosuite 1.4.0 code | Failure on mujoco 3.14 | Replacement |
|---|---|---|
| `MjModel.get_joint_qpos_addr`, `get_joint_qvel_addr`: `assert joint_type in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)` | a numpy integer never matches the 3.x pybind11 enums in a tuple membership test, so env construction raises AssertionError at the first hinge/slide joint | the same check on ints |
| `Controller.update`: `mujoco.mj_fullM(model, mass_matrix, data.qM)` | the 3.x signature is `mj_fullM(m, d, dst)` and `MjData` has no `qM` | `mujoco.mj_fullM(model, data, mass_matrix)` |

`robosuite_compat.apply()` replaces these three methods with copies that differ from robosuite 1.4.0 only in the lines
above (robosuite MIT notice kept in the file). It always runs, and each fix is applied only when the installed mujoco
needs it. No installed package file is modified.

## 3. Why the LIBERO data does not fit mujoco 3.x

After `set_init_state` and only the standard wait (`[0]*6+[-1]` for 10 steps), the original init states behave
differently on mujoco 3.14 than on mujoco 2.3.7:

| Measurement (50 original init states) | mujoco 2.3.7 | mujoco 3.14 |
|---|---|---|
| `libero_spatial` task 5: bowl-ramekin xy distance > 5 cm | 0/50 (median 1.45 cm) | 50/50 (median 6.23 cm) |
| `libero_10` tasks 0/1/7 (LIVING_ROOM 1/2): median of the largest object rise | 0.0292 m | 0.5737 m |
| 290 tasks x inits (10,900 pairs): pairs differing from 2.3.7 by > 1 mm / > 1 cm | - | 1,460 / 1,273 |

The init states were saved right after placement sampling, without settling (`notebooks/generate_init_states.py`),
so objects start up to ~3 cm inside the table or inside each other. All LIBERO object and fixture collision geoms are
boxes. mujoco 2.3.x reported half the true box-box penetration depth on the face path, which hid the overlap. Five
mujoco changes since 2.3.7 alter LIBERO results:

| Change | Version / commit | Effect |
|---|---|---|
| box-box face-path depth `points[i][2]` -> `2*points[i][2]` (bug fix) | 3.4.0 `88383684` | init objects sunk 1.5-2.9 cm into the table get twice the depth and launch. 93.6% of penetrating box-box pairs are exactly 2.000x (sample of 5,190 demo states) |
| box-box SAT collider rewrite, degenerate and deep-penetration fixes | 3.12.0 `86e98601`, `8655446f`, `fb07a9ca` | contact counts and positions change |
| convex collision: nativeccd default / multiccd default | 3.3.0 `ed16f2da` / 3.8.0 `6cb6e5a9` | gripper (mesh) to object (box) contacts 3,963 -> 6,902 (sample of 5,190 demo states) |
| Newton termination (decrement criterion) | 3.11.0 `1e66efd1` | at the default tolerance 1e-8 the solver stops one iteration earlier than 2.3.7 (about 30% of sampled demo states) |
| quaternion normalization: 2.3.7 `mj_kinematics` normalizes qpos quaternions in place, unconditionally; 3.x `mju_normalize4` skips \|norm-1\| <= 1e-15 and normalizes a copy only | 3.x | relevant only to the compat mode (§5): with the 2.3.7 box collider, last-bit differences in geom_xmat produce deep spurious contacts |

Model compiler changes (mesh inertia etc.) do not affect the physics: mass, inertia, friction, solref, solimp and
actuators agree within 1e-8 relative over 60 models.

## 4. Default: native mujoco 3.14 physics

### 4.1 Init states settled for mujoco 3.x (`libero/libero/init_files_mj314/`)

`scripts/settle_init_states.py` (core in `libero/libero/utils/settle_utils.py`) processes every init state of the
16 LIBERO-PRO suites (`libero_{spatial,object,goal,10}_{lan,object,swap,task}`, 160 tasks x 50):

1. `env.reset()`, `env.set_init_state(original)`;
2. fixture pose: `env.reset()` places the fixtures (stove, cabinets, wine rack, ...) at a random pose within a 2 cm
   region by writing `model.body_pos`/`body_quat`, and the original states do not record the pose they were sampled
   with. The script tries the pose from the reset and 15 more drawn from the same fixture samplers and keeps the one
   with the least penetration between objects and fixtures;
3. lift: each movable object that penetrates something below it is moved up by the penetration depth until no
   supporting contact is deeper than 0.1 mm; of two movable objects that penetrate each other, the one with the
   higher center of mass is moved;
4. lower: each movable object that rests on nothing is moved straight down to the first contact (lowest objects
   first). The original states hold many objects 6-16 cm above their support; dropped from there they bounce, tip or
   slide off (the bowl on the ramekin slides 5.6 cm off in every original init);
5. zero all velocities and run the standard wait action until every object moves less than 0.1 mm/s for 20 consecutive
   steps (1 s; 10 to 400 steps). With a looser criterion (1 mm/s for 5 steps) 7 inits with a bowl leaning in a drawer
   passed while still creeping, and the bowl fell flat 100-200 steps later;
6. the init is stable if the scene came to rest, the lift converged within 5 cm, no object moved more than 1 cm in xy
   while settling, no contact deeper than 1 cm remains (static interlocks of a few mm, e.g. a stove burner box inside a bowl's bottom
   boxes, stay at rest and are kept), and no object that touches another movable object ends tilted by more than
   10 degrees (a bowl leaning on a plate rim; under mujoco 2.3.7 these bowls start flat because the half-depth
   contact lets them overlap the rim). Objects tilted against a fixture or the table are kept: in `libero_goal_swap`
   tasks 6 and 7 the original layout puts the plate partly on the stove, and the plate leans on it under 2.3.7 too;
7. an unstable init is replaced by a new placement from the LIBERO sampler (with the fixture pose of that reset),
   settled the same way (up to 20 attempts); a task with an init for which no stable placement is found would get a
   `<file>.excluded` marker instead of an init file (none in this release);
8. the saved state is `[time, qpos, qvel]` followed by the pose (pos xyz, quat wxyz) of every fixture root body,
   sorted by fixture name (`ControlEnv.get_fixture_poses()`), so the arrays have shape `(50, 1 + nq + nv + 7 x
   n_fixtures)`.

`ControlEnv.set_init_state()` (and `regenerate_obs_from_state()`) restores the stored fixture poses, so every
`env.reset()` + `set_init_state()` reproduces the settled scene regardless of the random fixture pose the reset drew.
All LIBERO / LIBERO-PRO evaluation paths, `SubprocVectorEnv`, and the openvla-oft and openpi LIBERO harnesses load
init states this way. Code that calls `env.sim.set_state_from_flattened()` directly ignores the extra values (robosuite
reads only time, qpos and qvel) and keeps the fixture pose of the reset; objects that rest against a fixture can then
be pushed by up to tens of cm in the first steps.

Result (mujoco 3.14.0):

| Item | Value |
|---|---|
| Tasks | 160 saved, 0 excluded |
| Init states | 8,000; 7,877 keep the original layout, 123 (1.5%) replaced by a resampled placement (84 for an object tilted on another object, 30 for xy motion, 6 not at rest after 400 steps and tilted, 3 xy motion and tilted) |
| Most resampled tasks | `libero_spatial_lan` 0: 16, `libero_spatial_object` 0: 16, `libero_spatial_task` 0: 13, `libero_spatial_object` 2: 9, `libero_goal_swap` 5: 8 (bowl and plate overlap in the original layout) |
| xy shift while settling (original layouts) | median 0.000 mm, p99 2.24 mm, max 9.82 mm |
| Lowered onto support | 4,652 of 7,877 original-layout inits have an object lowered by more than 5 cm (max 16.3 cm) |
| Lift | max 3.8 cm, at most 3 lift iterations |
| Wait steps until at rest | median 22, max 375 |
| Remaining contact depth | median 0.015 mm, p99 0.035 mm, max 6.3 mm (`libero_10_object` task 7) |
| Fixture pose other than the reset's | 19 inits |
| Objects tilted > 10 degrees after settling | 87 inits: 81 in `libero_goal_swap` 6/7 (plate on the stove edge), 5 in `libero_10_swap` 2 (frying pan on the stove), 1 in `libero_goal_swap` 5; see §4.3 |
| Standard 10-step wait after loading | see §4.3 |

Known difference from the original start scene: in `libero_spatial_*` task 5 ("pick up the black bowl on the ramekin")
the original states hold the bowl 7.8 cm above the ramekin. Under mujoco 2.3.7 the bowl drops during the wait and comes
to rest leaning 15.9 degrees, 1.45 cm off center (the same in every init, since the bowl is placed relative to the
ramekin). Dropped under native 3.14 it slides 5.6 cm off the ramekin. The settled states have it flat and centered on
the ramekin.

Each init file has a `.report.json` with the settle parameters, the fixture names, and per init the lift, drop, xy/z
shift and tilt per object, the fixture candidate chosen, the remaining penetration and the source (`original` or
`resampled:<attempt>`, with the report of the rejected original).

How the benchmark picks init states (`Benchmark.get_task_init_states`):

- `LIBERO_INIT_STATES=original` or `mj314` selects explicitly.
- Otherwise `init_files_mj314` is used when mujoco >= 3.4 and the compat patch is not installed in the process.
- For a task with an `.excluded` marker it raises `benchmark.InitStatesUnavailable` with the reason.
- Suites without settled files (the four original LIBERO suites, libero_90) fall back to the original init states with a
  one-time warning; they keep the problems in §3 under native physics.

Reproduce: `LIBERO_INIT_STATES=original .venv/bin/python scripts/settle_init_states.py --suites <suite>` (all 16
suites: about 25 min with one process per suite; byte-identical output for a given mujoco version, checked on 3 tasks).

### 4.2 Demos regenerated for mujoco 3.x (`scripts/regenerate_demos_mj314.py`)

The LIBERO demos (`yifengzhu-hf/LIBERO-datasets`, made with `scripts/create_dataset.py`) were recorded and rendered with
mujoco 2.3.x. For each demo the script:

1. rebuilds the env from the demo's `model_file` (as `create_dataset.py` does, so the fixture poses are the recorded
   ones) and sets the demo's first state;
2. settles it as in §4.1 (lift, lower, wait) without the tilt test, because the recorded starts already contain objects
   that came to rest tilted under 2.3.x (the bowl on the ramekin is tilted 23.6 degrees in all 50 `libero_spatial`
   demos); then puts the robot joints back to the recorded start (the wait moves the arm by a median 0.024 rad, up to
   0.12 rad, and the recorded actions were made from the recorded pose); demos whose start is unstable are dropped;
3. reloads the settled state the way a user replays a demo (`env.reset()`, `reset_from_xml_string(model_file)`,
   `sim.reset()`, `sim.set_state_from_flattened(states[0])`, `sim.forward()`), so no controller or solver warm-start
   state carries over from settling;
4. replays the recorded actions open loop under native mujoco 3.14 and records the same fields as `create_dataset.py`
   (`states[t]` before action t, observations after it; 128x128 `agentview_rgb` / `eye_in_hand_rgb`, proprioception);
5. keeps the demo if the task is successful after the last action.

The output has the original hdf5 layout plus a `regenerated_with` attribute, per-demo `source_demo` and `fixture_poses`
attributes, and a per-task `.report.json`. A demo replays exactly through `model_file` as in step 3 (asset paths in
`model_file` are absolute paths of the machine that wrote it; `postprocess_model_xml` maps them to the local
installation). In an env built from the bddl file, `env.reset(); env.set_init_state(np.concatenate([states[0],
fixture_poses]))` restores the recorded fixture poses; with `set_init_state(states[0])` alone the fixtures keep the
random pose of the reset.

Result for the four original suites (the datasets are not in this repository; 26 GB):

| Suite | Demos kept | Dropped |
|---|---|---|
| libero_spatial | 442/500 (88.4%) | replay not successful 56, start not at rest after 400 steps 2 |
| libero_object | 452/500 (90.4%) | replay not successful 48 |
| libero_goal | 445/500 (89.0%) | replay not successful 55 |
| libero_10 | 370/500 (74.0%) | replay not successful 130 |
| total | 1,709/2,000 (85.5%) | 291 |

Open-loop replay of the original demos is not fully reliable on mujoco 2.3.7 itself. On a sample of 400 demos (4 suites
x 10 tasks x 10 demos) 345 (86.3%) succeed on 2.3.7 and 342 are kept by the regeneration (McNemar exact p = 0.80). The
outcome differs for 61 individual demos (32 succeed only on 2.3.7, 29 only after regeneration); on 2.3.7 a 1e-9
perturbation of the start state flips 4-8 of the 400.

Reproduce: `.venv/bin/python scripts/regenerate_demos_mj314.py --src <LIBERO-datasets> --dst <out>` (about 70 min with
8 processes, split by task).

### 4.3 Verification

All on mujoco 3.14.0, Python 3.14.4, unmodified robosuite 1.4.0.

| Check | Result |
|---|---|
| Reload stability: all 160 tasks x 50 inits, `env.seed(s); env.reset(); env.set_init_state(state)`, standard wait steps; seed 1 with 200 steps, seed 7 with 10 steps (16,000 resets) | largest free-object motion after 10 / 50 / 200 steps: 0.046 / 0.20 / 0.70 mm (seed 1), 0.046 mm after 10 steps (seed 7); 0 resets over 1 mm; fixture poses after `set_init_state` equal the stored ones (max error 0.0) |
| Same with the fixture pose of the reset instead of the stored one (states without the fixture poses, earlier version of this data) | seed 1: 118 inits over 1 mm, 7 over 5 cm, max 0.376 m |
| Original init states, native 3.14, same wait | 1,460 / 1,273 of 10,900 task-init pairs differ from 2.3.7 by > 1 mm / > 1 cm (§3) |
| Determinism | two runs of `settle_init_states.py` byte-identical (3 tasks); two runs of the demo generator identical in states, actions, images, `model_file` and `fixture_poses` (5 demos) |
| Object tilt against the 2.3.7 start (original init + 10 wait steps under real mujoco 2.3.7), 11 tasks with tilted objects, 472 original-layout inits | inits with an object tilted > 10 degrees: 86 settled, 91 on 2.3.7; per-init difference > 10 degrees: 11; tilted > 10 degrees only after settling (<= 5 on 2.3.7): 5. Without the tilt test: 64 and 48 |
| `libero_spatial_*` task 5 (bowl on the ramekin), 4 suites x 50 inits | all original layouts kept, 0 resets over 1 mm |
| Position against the 2.3.7 start, 10 inits per task, 1,577 original-layout inits (independent check of the previous version of this data, 1 mm/s rest criterion) | 63 inits differ by > 1 cm. In 58 inits 2.3.7 itself moves an object more than 1 cm from the original layout during the wait (bowls pushed 1-7 cm off plate rims, a wine bottle launched 2.6 m in `libero_10_task` 3); the settled states never do |
| Demo replay through `model_file` (fresh env, 24 demos from 8 files, 4 suites) | states and final agentview frame bitwise equal to the file, 24/24 successful |
| Demo replay in a bddl-built env with `set_init_state(states[0] + fixture_poses)` (same 24 demos) | start state equal to `states[0]`; 24/24 successful; the later states are not bitwise equal (cause not investigated) |
| `scripts/check_mujoco_compat.py` | native: all checks PASS; `LIBERO_MUJOCO_COMPAT=1`: all checks match mujoco 2.3.7 |

## 5. Optional: mujoco 2.3.7 compat mode (`LIBERO_MUJOCO_COMPAT=1`)

For comparison with published numbers, `LIBERO_MUJOCO_COMPAT=1` installs `libero/libero/envs/mujoco_compat` in the
process and the benchmark then uses the original init states. `install()`:

1. writes the address of a port of the 2.3.7 `mjc_BoxBox` (`compat_BoxBox_old`) into the writable data symbol
   `mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX]` of `libmujoco.so` via ctypes. The port is mujoco 2.3.7
   `engine_collision_box.c` lines 596-1332, verbatim except for renames (Apache-2.0), plus the 2.3.7 driver's removal of
   contacts at identical positions; compiled with `-O2 -ffp-contract=off` (with FMA, bit agreement with 2.3.7 drops to
   47.9% over 2,000 random box-box configurations);
2. wraps robosuite `MjSim.__init__` to set `opt.disableflags |= mjDSBL_NATIVECCD | mjDSBL_MULTICCD` and
   `opt.tolerance = 1e-12`;
3. wraps `mujoco.mj_step`, `mj_forward` and `mj_step1` to apply the 2.3.7 in-place quaternion normalization first.

`LIBERO_MUJOCO_COMPAT`: unset, `0`, `false`, `off`, `no` = native (default); `1`, `on`, `true`, `yes` = compat; `force` =
compat also on mujoco newer than the allowed range (3.9.0-3.14.x; only 3.14.0 was validated). Other values warn and mean
native. `mujoco_compat.status()` reports the state.

This mode restores a mujoco 2.3.7 bug (half box-box depth) on purpose, because the LIBERO data was made with it; report
results from it as "mujoco 3.14 + 2.3.7 compat".

Verification against mujoco 2.3.7 + robosuite 1.4.0 + Python 3.10 (noise baseline: 2.3.7 with +-1e-9 perturbed init
states):

| Item | Result |
|---|---|
| Contact lists: 200 tasks x 50 inits = 10,000 states | 10,000/10,000 bit-identical (412,364 contacts); qpos after forward bit-identical |
| 10-step init settle: 290 tasks, 10,900 pairs, > 1 mm / > 1 cm / > 5 cm | compat 1 / 0 / 0 (max 6.35 mm); 2.3.7 noise 4 / 3 / 0; native 3.14 with original inits 1,460 / 1,273 / 942 |
| Open-loop demo replay, 400 demos: success rate | compat 85.75%, 2.3.7 86.25% (McNemar p = 0.69), native 86.00% |
| Final object position difference, median / p90 | compat 0.22 / 5.61 mm; 2.3.7 noise 0.00-0.01 / 3.3-4.1 mm; 2.3.7 perturbed by 1e-6: 0.25 / 5.90 mm |
| Unmodified robosuite 1.4.0 + `robosuite_compat` | settle and replay bit-identical to the runs above |
| Speed | median `env.step` +2-6% |

The remaining differences come from parts the patch does not touch: from a bit-identical state and contact list one
forward differs in qacc by 5.4e-13 (solver arithmetic), and model compilation moves robot and gripper collision-mesh
vertices by up to 9e-9 m.

## 6. Notes

- `robosuite_compat` and `mujoco_compat` are per process. On Linux, Python 3.14's default start method is forkserver;
  `SubprocVectorEnv` workers apply them again when they import `libero.libero.envs`.
- `set_init_state` does not reset the solver warmstart, so the last bits can depend on the steps run before, for the same
  init state. Run comparisons in the same order (`env.reset()` -> `set_init_state` -> steps).
- Fixture poses: `env.reset()` samples them (within a 2 cm region for the stove, cabinets and wine rack). The original
  init states do not store them, so with the original data objects next to a fixture can start inside it (in the
  original LIBERO as well). The `init_files_mj314` states store them and `set_init_state` restores them (§4.1).
- Compat mode: every model stepped through `mujoco.mj_step/mj_forward/mj_step1` in the process gets its quaternions
  normalized in place; the 3.x driver reserves 8 contact slots per box-box pair (2.3.7 returns 9 for equal boxes stacked
  at a non-90-degree yaw; LIBERO maximum seen: 8, no overflow); the port keeps 2.3.7's bugs.
- Rendering: for the same sim state, mujoco 3.14 renders the `libero_object` floor much darker than mujoco 2.3.7 (mean
  absolute difference of the 128x128 agentview frame against the recorded 2.3.7 observation: about 39 of 255 under 3.14,
  about 3 under 2.3.7; the other three suites: 0.02-1.5 under both). Policies trained on 2.3.7 images see a different
  floor in `libero_object` under 3.14, in both the native and the compat mode (the compat patch changes only contacts).
  In the compiled 3.14 model the floor texture (`texplane`) has `tex_colorspace` sRGB (with the default
  `colorspace="auto"`, mujoco reads the sRGB chunk of `light-gray-floor-tile.png`); mujoco 2.3.7 has no colorspace
  handling. The `libero_object` floor fills most of the agentview image; in the other suites the table covers it. This
  fork does not change it. The regenerated demos are rendered by 3.14 and match what the policy sees at evaluation.
- Not checked: closed-loop policy success, mujoco 3.9-3.13 wheels.

## 7. Known LIBERO-PRO issues (verified, not fixed in this fork)

- **Instructions of the `_lan` and `_task` suites**: `task.language` is built from the bddl file name, and the PRO files
  keep the original names, so all 80 tasks return the original sentence. The perturbed sentence exists only in
  `env.language_instruction` (ControlEnv). The openvla-oft and openpi harnesses use `task.language`. The training demos
  also use the file-name sentence, so use `env.language_instruction` only for `_lan` and `_task`. The inner
  `env.env.language_instruction` raises KeyError (an original LIBERO bug).
- **`_env` suites**: there are no bddl/init files. `perturbation.py:598` writes `<suite>_temp`, not `<suite>_env`, and the
  seed default at `:658` is the `int` type object, which raises TypeError on Python 3.11+. `evaluation_config.yaml`
  defaults to `use_environment: true`.
- **HF robustness cases 01-07**: master prints a warning and ignores `:perturbation_config`. Cases 01-04, 06 and 07 equal
  the original tasks; case 05 adds an object, so loading the shared init state raises ValueError. The parser exists only
  on the unmerged upstream branch `codex/pr1-bddl-robustness`.
- **Missing custom_assets**: the xml files for `red_sticker`, `blue_red_sticker`, `red_box` and `libero_mug_green` are
  missing, so 24 older suites (`*_with_*` and others) fail at env construction (not part of the README/paper evaluation).
- **Data defects**: `libero_goal_task` task 7 starts with the stove already off (success needs knob qpos < 0).
  `libero_goal_swap` task 5 has the bowl and plate overlapping in 21/50 original inits. `libero_10_object` tasks 9 and 5
  name objects that are not in the scene. `libero_10_task` task 8 has left/right reversed.
- **`task_order_index` != 0**: only the original suites are reordered, the PRO suites are not, so the same index points
  at different tasks.
- **lifelong training code**: imports fail with robomimic 0.3, and three `torch.load` calls are unchanged. Its evaluation
  (`libero/lifelong/metric.py`, `evaluate.py`) reads the original init files directly, not through
  `get_task_init_states`, so it would not use `init_files_mj314`. This does not affect the VLA evaluation path.
