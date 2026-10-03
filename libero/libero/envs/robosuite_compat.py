"""robosuite 1.4.0 fixes for the mujoco 3.x Python bindings.

robosuite 1.4.0 was written against mujoco 2.3.x. Two code paths that LIBERO uses fail on mujoco 3.14:

1. MjModel.get_joint_qpos_addr / get_joint_qvel_addr assert
   `joint_type in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)`. With the 3.x pybind11 enums a numpy
   integer never matches inside a tuple membership test, so env construction fails on the first
   hinge or slide joint (AssertionError).
2. Controller.update builds the mass matrix with mj_fullM(m, dst, d.qM). In 3.x the signature is
   mj_fullM(m, d, dst) and MjData has no qM attribute.

apply() replaces these three methods with versions that differ from robosuite 1.4.0 only in those
lines. It is independent of LIBERO_MUJOCO_COMPAT: without it LIBERO cannot build an env on mujoco 3.x.
Each fix is applied only when the installed mujoco needs it.

The replacement methods are adapted from robosuite 1.4.0 (robosuite/utils/binding_utils.py and
robosuite/controllers/base_controller.py), which carries this notice:

    MIT License

    Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import mujoco
import numpy as np

_MARK = "_libero_robosuite_compat"
_HINGE_SLIDE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def _get_joint_qpos_addr(self, name):
    """robosuite 1.4.0 MjModel.get_joint_qpos_addr, with the hinge/slide check done on ints."""
    joint_id = self.joint_name2id(name)
    joint_type = self.jnt_type[joint_id]
    joint_addr = self.jnt_qposadr[joint_id]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert int(joint_type) in _HINGE_SLIDE
        ndim = 1

    if ndim == 1:
        return joint_addr
    else:
        return (joint_addr, joint_addr + ndim)


def _get_joint_qvel_addr(self, name):
    """robosuite 1.4.0 MjModel.get_joint_qvel_addr, with the hinge/slide check done on ints."""
    joint_id = self.joint_name2id(name)
    joint_type = self.jnt_type[joint_id]
    joint_addr = self.jnt_dofadr[joint_id]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 3
    else:
        assert int(joint_type) in _HINGE_SLIDE
        ndim = 1

    if ndim == 1:
        return joint_addr
    else:
        return (joint_addr, joint_addr + ndim)


def _controller_update(self, force=False):
    """robosuite 1.4.0 Controller.update, with the mujoco 3.x mj_fullM(m, d, dst) call."""
    # Only run update if self.new_update or force flag is set
    if self.new_update or force:
        self.sim.forward()

        self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])
        self.ee_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_name)].reshape([3, 3]))
        self.ee_pos_vel = np.array(self.sim.data.get_site_xvelp(self.eef_name))
        self.ee_ori_vel = np.array(self.sim.data.get_site_xvelr(self.eef_name))

        self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
        self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

        self.J_pos = np.array(self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_ori = np.array(self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

        mass_matrix = np.ndarray(shape=(self.sim.model.nv, self.sim.model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]

        # Clear self.new_update
        self.new_update = False


def _enum_membership_broken():
    return np.int32(_HINGE_SLIDE[0]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


def _fullm_takes_data():
    return not hasattr(mujoco.MjData, "qM")


def apply():
    """Apply the robosuite fixes the installed mujoco needs. Returns the list of applied fixes."""
    from robosuite.controllers import base_controller
    from robosuite.utils import binding_utils

    applied = []
    model_cls = binding_utils.MjModel
    if _enum_membership_broken() and not model_cls.__dict__.get(_MARK):
        model_cls.get_joint_qpos_addr = _get_joint_qpos_addr
        model_cls.get_joint_qvel_addr = _get_joint_qvel_addr
        setattr(model_cls, _MARK, True)
        applied.append("joint_addr")
    controller_cls = base_controller.Controller
    if _fullm_takes_data() and not controller_cls.__dict__.get(_MARK):
        controller_cls.update = _controller_update
        setattr(controller_cls, _MARK, True)
        applied.append("mj_fullM")
    return applied
