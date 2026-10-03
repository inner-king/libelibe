"""Settle a LIBERO scene under the installed mujoco physics while keeping the object layout.

Used by scripts/settle_init_states.py (evaluation init states) and scripts/regenerate_demos_mj314.py
(demo start states). The original LIBERO states were saved right after placement sampling, without
settling, for mujoco 2.3.x, whose box-box collider reported half the penetration depth. Objects start
up to ~3 cm inside the table or each other, or 6-16 cm above their support. Under mujoco >= 3.4 the
penetrations launch or push objects, and the dropped objects bounce or slide off their support.

settle_current():
1. lift: every movable object that penetrates something below it is moved up by the penetration
   depth, repeated until no supporting contact is deeper than pen_tol. When two movable objects
   penetrate each other, the one whose center of mass is higher is moved;
2. lower: every movable object that rests on nothing is moved straight down until it touches
   something below it (lowest objects first, repeated until nothing moves);
3. zero all velocities and run the standard wait action ([0]*6 + [-1]) until every object moves less
   than speed_tol for quiet_steps consecutive steps (at least min_steps, at most max_steps; a scene
   that is still moving at max_steps is unstable);
4. report per-object lift, drop, xy/z shift and tilt (angle between the body z axes before step 1
   and after step 3). `stable` = the scene came to rest, the lift converged within max_lift, max xy
   shift <= max_dxy, the
   deepest remaining contact <= max_pen, and no object that touches another movable object is tilted
   by more than max_tilt (a bowl left leaning on a plate rim; under mujoco 2.3.7 these bowls start
   flat because the soft half-depth contact lets them overlap the rim). Objects tilted against a
   fixture or the table are allowed: where the original layout puts a plate partly on the stove, the
   plate also leans on the stove under mujoco 2.3.7.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

WAIT = np.array([0.0] * 6 + [-1.0])


@dataclass
class SettleParams:
    pen_tol: float = 1e-4  # m; deeper supporting contacts are lifted
    speed_tol: float = 1e-4  # m/s; stop criterion (a bowl leaning in a drawer creeps at ~1 mm/s and falls later)
    quiet_steps: int = 20  # consecutive steps below speed_tol (1 s at 20 Hz)
    min_steps: int = 10
    max_steps: int = 400
    max_dxy: float = 0.01  # m; larger xy motion while settling = unstable
    max_pen: float = 1e-2  # m; deeper remaining contact = unstable (static interlocks of a few mm, e.g. a
    # burner box inside a bowl's bottom boxes, are at rest and allowed)
    max_lift: float = 0.05  # m; an object lifted further, or a lift that does not converge = unstable
    max_tilt: float = 10.0  # deg; larger tilt of an object touching another movable object = unstable
    max_drop: float = 0.3  # m; how far below a floating object a support is searched for


def free_object_bodies(model):
    """Map body id -> free joint qpos address for every body that owns a free joint."""
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            out[int(model.jnt_bodyid[j])] = int(model.jnt_qposadr[j])
    return out


def owner(model, body, free):
    """The free-joint body that a body belongs to (walking up the tree), or -1."""
    while body > 0:
        if body in free:
            return body
        body = int(model.body_parentid[body])
    return -1


def _contacts(model, data, free):
    """(b1, b2, dist, nz) for contacts between different owners; b = -1 for fixtures, robot and world.
    The contact normal points from geom1 (b1) to geom2 (b2)."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = owner(model, int(model.geom_bodyid[c.geom1]), free)
        b2 = owner(model, int(model.geom_bodyid[c.geom2]), free)
        if b1 != b2:
            yield b1, b2, float(c.dist), float(c.frame[2])


def lift_penetrations(model, data, free, pen_tol, max_iter=50):
    """Lift objects out of supporting contacts. Returns (iterations, lifted per body, side contacts)."""
    lifted = {b: 0.0 for b in free}
    side = 0
    for it in range(max_iter):
        mujoco.mj_forward(model, data)
        need = {}
        side = 0
        for b1, b2, dist, nz in _contacts(model, data, free):
            if dist >= -pen_tol:
                continue
            if abs(nz) < 0.5:
                side += 1
                continue
            if b1 >= 0 and b2 >= 0:  # two movable objects: move the one whose center of mass is higher
                top = b1 if data.xipos[b1][2] > data.xipos[b2][2] else b2
            else:  # object and fixture/world: move the object only if it is the upper side
                top = b2 if nz > 0 else b1
                if top < 0:
                    continue
            need[top] = max(need.get(top, 0.0), -dist / abs(nz))
        if not need:
            return it, lifted, side
        for b, dz in need.items():
            data.qpos[free[b] + 2] += dz + 1e-4
            lifted[b] += dz + 1e-4
    mujoco.mj_forward(model, data)
    return max_iter, lifted, side


def _supported(model, data, free, body, pen_tol):
    """True if `body` touches something below it (a contact with |nz| >= 0.5 and `body` on top)."""
    for b1, b2, dist, nz in _contacts(model, data, free):
        if dist >= pen_tol or abs(nz) < 0.5:
            continue
        if (b2 == body and nz > 0) or (b1 == body and nz < 0):
            return True
    return False


def lower_floating(model, data, free, pen_tol, max_drop, step=2e-3, tol=1e-5, max_pass=10):
    """Move every unsupported object straight down to the first contact below it.
    Returns (dropped per body, bodies with no support within max_drop)."""
    dropped = {b: 0.0 for b in free}
    no_support = set()
    for _ in range(max_pass):
        mujoco.mj_forward(model, data)
        moved = False
        for b in sorted(free, key=lambda b: data.xipos[b][2]):
            mujoco.mj_forward(model, data)
            if _supported(model, data, free, b, pen_tol):
                continue
            adr = free[b] + 2
            z0 = float(data.qpos[adr])
            lo, hi = 0.0, None  # lo: largest drop known free, hi: smallest drop known in contact
            d = step
            while d <= max_drop + 1e-12:
                data.qpos[adr] = z0 - d
                mujoco.mj_forward(model, data)
                if _supported(model, data, free, b, pen_tol):
                    hi = d
                    break
                lo = d
                d += step
            if hi is None:
                data.qpos[adr] = z0
                no_support.add(b)
                continue
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                data.qpos[adr] = z0 - mid
                mujoco.mj_forward(model, data)
                if _supported(model, data, free, b, pen_tol):
                    hi = mid
                else:
                    lo = mid
            data.qpos[adr] = z0 - lo
            dropped[b] += lo
            moved = moved or lo > tol
        if not moved:
            break
    mujoco.mj_forward(model, data)
    return dropped, no_support


def max_penetration(model, data, free):
    """Deepest contact between different bodies that involves a movable object (m, >= 0)."""
    deepest = 0.0
    for b1, b2, dist, _ in _contacts(model, data, free):
        if b1 >= 0 or b2 >= 0:
            deepest = max(deepest, -dist)
    return deepest


def _tilt_deg(z0, z1):
    return float(np.degrees(np.arccos(np.clip(np.dot(z0, z1), -1.0, 1.0))))


def settle_current(env, params):
    """Settle the scene currently loaded in `env` (a LIBERO ControlEnv). Returns a report dict."""
    sim = env.env.sim
    model, data = sim.model._model, sim.data._data
    free = free_object_bodies(model)
    mujoco.mj_forward(model, data)
    p0 = {b: data.xpos[b].copy() for b in free}
    z0 = {b: data.xmat[b].reshape(3, 3)[:, 2].copy() for b in free}

    iters, lifted, side = lift_penetrations(model, data, free, params.pen_tol)
    dropped, no_support = lower_floating(model, data, free, params.pen_tol, params.max_drop)
    data.qvel[:] = 0.0
    if data.act is not None and len(data.act):
        data.act[:] = 0.0
    mujoco.mj_forward(model, data)

    quiet, steps, speed = 0, 0, 0.0
    prev = {b: data.xpos[b].copy() for b in free}
    dt = 1.0 / env.env.control_freq
    while steps < params.max_steps:
        env.step(WAIT)
        steps += 1
        speed = max((np.linalg.norm(data.xpos[b] - prev[b]) / dt for b in free), default=0.0)
        prev = {b: data.xpos[b].copy() for b in free}
        quiet = quiet + 1 if speed < params.speed_tol else 0
        if steps >= params.min_steps and quiet >= params.quiet_steps:
            break
    at_rest = quiet >= params.quiet_steps
    names = {b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in free}
    mujoco.mj_forward(model, data)
    pen = max_penetration(model, data, free)
    dxy = {b: float(np.linalg.norm(data.xpos[b][:2] - p0[b][:2])) for b in free}
    tilt = {b: _tilt_deg(z0[b], data.xmat[b].reshape(3, 3)[:, 2]) for b in free}
    max_dxy = max(dxy.values(), default=0.0)
    max_lift = max(lifted.values(), default=0.0)
    lift_ok = iters < 50 and max_lift <= params.max_lift
    tilted_on_object = set()
    for b1, b2, _, _ in _contacts(model, data, free):
        if b1 >= 0 and b2 >= 0:
            tilted_on_object.update(b for b in (b1, b2) if tilt[b] > params.max_tilt)
    return {
        "stable": bool(at_rest and lift_ok and max_dxy <= params.max_dxy and pen <= params.max_pen
                       and not tilted_on_object),
        "at_rest": bool(at_rest),
        "max_dxy": round(max_dxy, 6),
        "max_pen_after": round(pen, 6),
        "max_lift": round(max_lift, 6),
        "max_drop": round(max(dropped.values(), default=0.0), 6),
        "max_tilt_deg": round(max(tilt.values(), default=0.0), 3),
        "lift_iters": iters,
        "no_support": sorted(names[b] for b in no_support),
        "tilted_on_object": sorted(names[b] for b in tilted_on_object),
        "steps": steps,
        "final_speed": float(speed),
        "side_contacts": side,
        "objects": {
            names[b]: {
                "lift": round(lifted[b], 6),
                "drop": round(dropped[b], 6),
                "dxy": round(dxy[b], 6),
                "dz": round(float(data.xpos[b][2] - p0[b][2]), 6),
                "tilt_deg": round(tilt[b], 3),
            }
            for b in free
        },
    }
