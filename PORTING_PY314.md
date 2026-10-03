# Running LIBERO-PRO on Python 3.14 / mujoco 3.14 with mujoco 2.3.7 physics

This repository is [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) master `eafdb80` (2026-03-23)
plus the changes below. It has two goals.

1. Run LIBERO-PRO evaluation on Python 3.14, torch 2.14 and mujoco 3.14.
2. LIBERO init states and demos were made with mujoco 2.3.x. On mujoco 3.14 the scene changes before the policy
   acts (a bowl slides off the ramekin, objects launch 0.57 m above the table). Python 3.14 has no mujoco wheel
   below 3.8.0, so mujoco cannot be downgraded. mujoco 3.14 is kept and the 2.3.7 physics is restored with a
   runtime patch.

---

## 1. Installation

```bash
git clone https://github.com/inner-king/libelibe.git && cd libelibe
uv venv --python 3.14.4 .venv
uv pip install --python .venv/bin/python -r requirements-py314.txt
uv pip install --python .venv/bin/python -e .
echo n | .venv/bin/python -c "import libero.libero"                  # creates the LIBERO path config (see below)
MUJOCO_GL=egl .venv/bin/python scripts/check_mujoco_compat.py    # all three checks must PASS
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
| Main packages | mujoco 3.14.0, robosuite 1.4.0, torch 2.14.1 (PyPI, CUDA 13), numpy 1.26.4, gym 0.26.2; full list in `requirements-py314.txt` |
| Build tool | gcc. `compat_237.c` is compiled on first import (under 1 s; measured 0.2-0.5 s). Without gcc the import raises RuntimeError |
| Rendering | `MUJOCO_GL=egl` |

The LIBERO-PRO evaluation bddl/init files are not in this repository. As in the upstream README, copy
`bddl_files/*` from the HF dataset [`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)
into `libero/libero/bddl_files/` and `init_files/*` into `libero/libero/init_files/`. The files of the four original
LIBERO suites (spatial/object/goal/10) and libero_90 are in git.

## 2. Changes against upstream

| File | Change | Reason |
|---|---|---|
| `libero/libero/benchmark/__init__.py:273` | `torch.load(..., weights_only=False)` | With the torch >= 2.6 default `weights_only=True`, loading the init files fails with UnpicklingError |
| `libero/libero/envs/env_wrapper.py` `ControlEnv.reset` | removed the 2-line `finally: continue` | Python 3.14 PEP 765 SyntaxWarning. Behaviour change: before, every exception (KeyboardInterrupt included) was swallowed and retried forever; now only `RandomizationError` is retried |
| `libero/__init__.py` (new, empty) | added | Without it the editable-install finder maps no package and `import libero` fails outside the repo root |
| `libero/libero/envs/__init__.py`, first 3 lines | calls `mujoco_compat.install()` | §3 |
| `libero/libero/envs/mujoco_compat/` (new) | `__init__.py`, `compat_237.c`, `LICENSE`, `.gitignore` | §3. `compat_237.c` is derived from mujoco 2.3.7 and is Apache-2.0; `LICENSE` is that license text (the rest of the repo is MIT). The build output `_compat_237-*.so` is git-ignored |
| `scripts/check_mujoco_compat.py` (new) | install check | §1, §4 |
| `requirements-py314.txt` (new) | `uv pip freeze` of the validated environment | §1 |

## 3. On mujoco 3.14 the LIBERO scene changes before the policy acts

### Symptoms

After `set_init_state` and only the standard wait (`[0]*6+[-1]` for 10 steps), object positions differ from 2.3.7.

| Measurement (50 init states) | mujoco 2.3.7 | mujoco 3.14 native |
|---|---|---|
| `libero_spatial` task 5: bowl-ramekin xy distance > 5 cm | 0/50 (median 1.45 cm) | 50/50 (median 6.23 cm) |
| `libero_10` tasks 0/1/7 (LIVING_ROOM 1/2): median of the largest object rise | 0.0292 m | 0.5737 m |
| 290 tasks x inits (10,900 pairs): pairs differing from 2.3.7 by > 1 mm / > 1 cm | - | 1,460 / 1,273 |

### Cause

All LIBERO object and fixture collision geoms are boxes (meshes are only on the robot and gripper, a cylinder on the
mount, capsules on the microwave handle). Five changes since 2.3.7 alter LIBERO results.

| Change | Version / commit | Effect |
|---|---|---|
| box-box face-path depth `points[i][2]` -> `2*points[i][2]` | 3.4.0 `88383684` | init objects sunk 1.5-2.9 cm into the table get twice the depth and launch. 93.6% of penetrating box-box pairs are exactly 2.000x (sample of 5,190 demo states) |
| box-box SAT collider rewrite, degenerate and deep-penetration fixes | 3.12.0 `86e98601`, `8655446f`, `fb07a9ca` | contact counts and positions change |
| convex collision: nativeccd default / multiccd default | 3.3.0 `ed16f2da` / 3.8.0 `6cb6e5a9` | gripper (mesh) to object (box) contacts 3,963 -> 6,902 (sample of 5,190 demo states) |
| Newton termination (decrement criterion) | 3.11.0 `1e66efd1` | at the default tolerance 1e-8 the solver stops one iteration earlier than 2.3.7 (about 30% of sampled demo states) |
| quaternion normalization: 2.3.7 `mj_kinematics` normalizes qpos quaternions in place, unconditionally; 3.x `mju_normalize4` skips \|norm-1\| <= 1e-15 and normalizes a copy only | 3.x | 11,890 of the 65,850 object quaternions in the init files (290 tasks x 50 inits) are affected. The last bits of geom_xmat change and the 2.3.7 box algorithm emits a deep spurious contact at a far corner (books in STUDY scenes launch 2.4 cm to 125.7 m) |

Halving the depth alone does not work: 2.3.7 halved only the face path and reported the full depth on the edge path
(halving everything leaves 5 pairs above 1 cm in a 40-task x 10-init settle comparison).
Model compiler changes (mesh inertia etc.) do not affect the physics: mass, inertia, friction, solref, solimp and
actuators agree within 1e-8 relative over 60 models.

### Fix: `libero/libero/envs/mujoco_compat/`

`install()` does the following once per process. Importing `libero.libero.envs` calls it.

1. Writes the address of the 2.3.7 `mjc_BoxBox` port (`compat_BoxBox_old`) into the writable data symbol
   `mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX]` of `libmujoco.so` via ctypes. The port is mujoco 2.3.7
   `engine_collision_box.c` lines 596-1332, verbatim except for renames (`mjContact`->`oldcon_t`, `mju_*`->`old_*`
   static copies; Apache-2.0). It reproduces the 2.3.7 driver's removal of contacts at identical positions and writes
   the result as 3.x `mjPreContact`. Compiler flags are `-O2 -ffp-contract=off` (with FMA, bit agreement with 2.3.7
   drops to 47.9% over 2,000 random box-box configurations).
2. Wraps robosuite `MjSim.__init__` to set `opt.disableflags |= mjDSBL_NATIVECCD | mjDSBL_MULTICCD` and
   `opt.tolerance = 1e-12` before the original constructor. robosuite builds a new `MjSim` on every hard reset, so
   this persists across `env.reset()`.
3. Wraps the mujoco module functions `mj_step`, `mj_forward` and `mj_step1` to call `compat_normalizeQuat_237`
   (a C port of 2.3.7 `mj_normalizeQuat` + `mju_normalize4`) first. All three run `mj_kinematics` in 2.3.7, and they are
   the only mujoco functions called by robosuite and LIBERO that do (`binding_utils.py:1088,1092`,
   `bddl_base_domain.py:755`).

The island solver stays on (3.14 default). Turning it off raises the largest difference from 2.3.7 over 40
zero-action steps from 6.5e-7 m to 1.3-1.8e-6 m (`libero_10` task 0, `libero_10_lan` task 0, `libero_10_swap` task 7).

### Usage

```bash
.venv/bin/python -c "from libero.libero.envs import mujoco_compat; print(mujoco_compat.status())"
# {'mujoco': '3.14.0', 'mode': 'on', 'boxbox_slot_is_compat': True, 'mjsim_wrapped': True, 'mujoco_wrapped': True, 'installed': True, ...}
LIBERO_MUJOCO_COMPAT=0 .venv/bin/python ...     # native mujoco 3.14 physics, no patch
```

- `LIBERO_MUJOCO_COMPAT`: unset, `''`, `1`, `on`, `true`, `yes` = on; `0`, `false`, `off`, `no` = off;
  `force` = also install on mujoco newer than the allowed range (3.9.0-3.14.x). Other values print a warning and mean on.
- Only mujoco 3.14.0 was validated. 3.9-3.13 are allowed because they have the same collision function interface and
  the same number of box-box contact slots (8); they were not run.
- Outside the range the patch is not installed; the reason is printed to stderr and physics is native. On mujoco 2.x it does nothing.
- The `.so` is built in the package folder, or in `~/.cache/libero/mujoco_compat/` if that is not writable. Its file
  name contains a hash of the source, the compiler flags and the mujoco version.

## 4. Verification (against mujoco 2.3.7 + robosuite 1.4.0 + Python 3.10)

The noise baseline is 2.3.7 with the init-state object positions (or qpos) perturbed by +-1e-9. Both versions give
bit-identical results when rerun on the same input.

| Item | Result |
|---|---|
| Contact lists: 4 base + 16 PRO suites, 200 tasks x 50 inits = 10,000 states | 10,000/10,000 bit-identical (412,364 contacts). qpos after forward also bit-identical in 10,000/10,000 |
| 10-step init settle: 290 tasks (incl. libero_90), 10,900 pairs, > 1 mm / > 1 cm / > 5 cm | compat 1 / 0 / 0 (max 6.35 mm); 2.3.7 noise 4 / 3 / 0 (max 15.4 mm); 3.14 native 1,460 / 1,273 / 942 (max 7.07 m) |
| The two symptoms above | `libero_spatial` task 5: 0/50, 1.45 cm; `libero_10` task 0 rise 0.0292 m. Same as 2.3.7 (`scripts/check_mujoco_compat.py`) |
| Open-loop demo replay, 400 demos (4 base suites x 10 tasks x 10 demos): success rate | compat 85.75%, 2.3.7 86.25% (McNemar p = 0.69), 3.14 native 86.00% |
| Demos whose success differs from 2.3.7 | compat 6; 2.3.7 perturbed by +-1e-9, 3 runs: 4 / 4 / 6; 3.14 native 39 |
| Final object position difference, median / p90 | compat 0.22 / 5.61 mm; 2.3.7 noise 0.00-0.01 / 3.3-4.1 mm; 2.3.7 perturbed by 1e-6: 0.25 / 5.90 mm; 3.14 native 5.19 / 44.55 mm |
| SubprocVectorEnv (forkserver / spawn / fork), DummyVectorEnv | worker results bit-identical to in-process results |
| Speed | median `env.step` +2-6% (`libero_10` task 0 5.52 -> 5.66 ms, `libero_spatial` task 0 7.62 -> 8.08 ms, pinned to one CPU core) |

The two remaining settle pairs (`libero_90` task 5 init 15: 6.35 mm; `libero_spatial_task` task 0 init 46: 0.38 mm)
are scenes where 2.3.7 itself diverges by the same amount under 1e-9 to 1e-8 perturbations.
Continuous final positions in demo replay are larger than the 1e-9 noise and at the level of a 1e-6 perturbation of
2.3.7. From a bit-identical state and contact list, one forward already differs in qacc by 5.4e-13 (solver and
dynamics arithmetic order), and model compilation moves robot and gripper collision-mesh vertices by up to 9e-9 m
(hand 7.7e-9 m, fingers 1.8e-9 m). The patch does not touch either.

## 5. Notes

- The patch is per process. On Linux, Python 3.14's default start method is forkserver, so `SubprocVectorEnv` workers
  do not inherit the parent's patch; it is applied again when the worker imports `libero.libero.envs`. Code that uses
  robosuite without LIBERO calls `mujoco_compat.install()` itself.
- Every model stepped through `mujoco.mj_step/mj_forward/mj_step1` in the process gets its qpos quaternions and
  mocap_quat normalized in place. Code that bound these functions before `install()` (`from mujoco import mj_step`) or
  calls `mj_kinematics`, `mj_fwdPosition` or `mj_forwardSkip` skips the normalization.
- The 3.x driver reserves 8 contact slots per box-box pair. 2.3.7 returns 9 when equal boxes are stacked face to face
  and the top one is rotated about the vertical axis by an angle that is not a multiple of 90 degrees (12 pairs in
  4,000 synthetic scenes). The port then keeps 8. In LIBERO the maximum seen is 8 per pair with 0 overflows
  (`boxbox_nmax`, `boxbox_noverflow` in `status()`).
- The port keeps 2.3.7's bugs. Example: when a resting box's tilt decays to about 1e-17, one extra spurious contact
  appears at the table corner (seen in demo replay of `libero_spatial` task 0 demo 0 and `libero_goal` task 0 demo 0).
  This event happening one or two steps apart in 2.3.7 and compat is the main source of the final-position differences.
- `set_init_state` does not reset the solver warmstart, so the last bits can depend on the steps run before, for the
  same init state (also in 2.3.7). Run comparisons in the same order (`env.reset()` -> `set_init_state` -> steps).
- Differences not reverted: mesh-plane collider rewrite (3.13.0), capsule-box (3.5.0), the second normalization inside
  2.3.7 `mj_kinematics` and the integrator normalization (last bits), solver-internal rounding, the renderer, the geom
  margin combination rule (2.3.7 takes the max of the two geoms, 3.5.0 `6af0d4c8` and later take the sum) and the
  margin/gap redefinition (3.9.0 `a4e49f2d`). In LIBERO scenes only contype=0 geoms such as the microwave have
  margin > 0 and every gap is 0, so LIBERO results are unaffected. Scenes where both geoms have margin > 0 get a
  different contact count than 2.3.7.
- Not checked: closed-loop policy success compared with 2.3.7, camera images rendered by mujoco 3.14 vs 2.3.7,
  mujoco 3.9-3.13 wheels.

## 6. Known LIBERO-PRO issues (verified, not fixed in this fork)

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
  `libero_goal_swap` task 5 has the bowl and plate overlapping in 21/50 inits. `libero_10_object` tasks 9 and 5 name
  objects that are not in the scene. `libero_10_task` task 8 has left/right reversed.
- **`task_order_index` != 0**: only the original suites are reordered, the PRO suites are not, so the same index points
  at different tasks.
- **lifelong training code**: imports fail with robomimic 0.3, and three `torch.load` calls are unchanged. This does not
  affect the VLA evaluation path.
