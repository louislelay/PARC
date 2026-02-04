"""
mjlab Character Environment for PARC Motion Tracking.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import gymnasium as gym
import mujoco
import mujoco_warp as mjwarp

# mjlab imports
from mjlab.envs import ManagerBasedRlEnv
from mjlab.actuator.actuator import Actuator, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator
from mjlab.entity import Entity

# PARC imports
import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.torch_util as torch_util

if TYPE_CHECKING:
    from parc.motion_tracker.envs.mj_env_config import MJCharEnvCfg, PDExpActuatorCfg

# =============================================================================
# Custom Actuators
# =============================================================================

class PDExpActuator(Actuator):
    """
    Custom actuator for exponential map PD control (Spherical Joints).
    """

    def __init__(
        self,
        cfg: PDExpActuatorCfg,
        entity: Any,
        target_ids: list[int],
        target_names: list[str],
    ):
        super().__init__(cfg, entity, target_ids, target_names)
        self.cfg: PDExpActuatorCfg = cfg
        self._kin_char_model: Optional[kin_char_model.KinCharModel] = None
        
        self._kp: Optional[torch.Tensor] = None
        self._kd: Optional[torch.Tensor] = None
        self._torque_lim: Optional[torch.Tensor] = None

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        """Register motors in the MuJoCo spec."""
        for target_name in target_names:
            actuator = create_motor_actuator(
                spec,
                target_name,
                effort_limit=self.cfg.effort_limit,
                armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,
                transmission_type=self.cfg.transmission_type,
            )
            self._mjs_actuators.append(actuator)

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        """Initialize on device."""
        super().initialize(mj_model, model, data, device)
        
        # Load Kinematic Model on device
        self._kin_char_model = kin_char_model.KinCharModel(device)
        if hasattr(self.cfg, 'kin_char_model_path') and self.cfg.kin_char_model_path:
             self._kin_char_model.load_char_file(self.cfg.kin_char_model_path)

        num_envs = data.nworld
        self._kp = self._parse_gains(self.cfg.stiffness, self.target_names, num_envs, device)
        self._kd = self._parse_gains(self.cfg.damping, self.target_names, num_envs, device)
        self._torque_lim = self._parse_gains(self.cfg.effort_limit, self.target_names, num_envs, device)

    def _parse_gains(self, gains: float | dict, joint_names: List[str], num_envs: int, device: str) -> torch.Tensor:
        num_joints = len(joint_names)
        if isinstance(gains, (int, float)):
            val = torch.full((num_envs, num_joints), gains, device=device)
        else:
            base = torch.zeros(num_joints, device=device)
            for i, name in enumerate(joint_names):
                match = False
                for pattern, value in gains.items():
                    if pattern in name:
                        base[i] = value
                        match = True
                        break
                if not match:
                    base[i] = 10.0
            val = base.unsqueeze(0).repeat(num_envs, 1)
        return val

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        if self._kin_char_model is None:
            raise RuntimeError("Actuator not initialized")

        sim_joint_rot = self._kin_char_model.dof_to_rot(cmd.pos)
        tar_joint_rot = self._kin_char_model.dof_to_rot(cmd.position_target)
        
        diff_dof = self._kin_char_model.compute_dof_vel(sim_joint_rot, tar_joint_rot, 1.0)
        
        torque = self._kp * diff_dof - self._kd * cmd.vel
        torque = torch.clamp(torque, -self._torque_lim, self._torque_lim)
        
        if cmd.effort_target is not None:
            torque += cmd.effort_target
            
        return torque


class PD1DActuator(Actuator):
    """Custom actuator for standard 1D joint PD control."""

    def __init__(
        self,
        cfg: Any,
        entity: Any,
        target_ids: list[int],
        target_names: list[str],
    ):
        super().__init__(cfg, entity, target_ids, target_names)
        self.cfg = cfg
        self._kp: Optional[torch.Tensor] = None
        self._kd: Optional[torch.Tensor] = None
        self._torque_lim: Optional[torch.Tensor] = None

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            actuator = create_motor_actuator(
                spec,
                target_name,
                effort_limit=self.cfg.effort_limit,
                armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,
                transmission_type=self.cfg.transmission_type,
            )
            self._mjs_actuators.append(actuator)

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        num_envs = data.nworld
        self._kp = self._parse_gains(self.cfg.stiffness, self.target_names, num_envs, device)
        self._kd = self._parse_gains(self.cfg.damping, self.target_names, num_envs, device)
        self._torque_lim = self._parse_gains(self.cfg.effort_limit, self.target_names, num_envs, device)

    def _parse_gains(self, gains, joint_names, num_envs, device) -> torch.Tensor:
        num_joints = len(joint_names)
        if isinstance(gains, (int, float)):
            val = torch.full((num_envs, num_joints), gains, device=device)
        else:
            base = torch.zeros(num_joints, device=device)
            for i, name in enumerate(joint_names):
                for pattern, value in gains.items():
                    if pattern in name:
                        base[i] = value
                        break
            val = base.unsqueeze(0).repeat(num_envs, 1)
        return val

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        torque = self._kp * (cmd.position_target - cmd.pos) - self._kd * cmd.vel
        torque = torch.clamp(torque, -self._torque_lim, self._torque_lim)
        if cmd.effort_target is not None:
            torque += cmd.effort_target
        return torque


# =============================================================================
# Main Environment Class
# =============================================================================

class MJCharEnv(ManagerBasedRlEnv):
    """
    mjlab Character Environment for PARC Motion Tracking.
    """

    def __init__(self, cfg: MJCharEnvCfg, render_mode: str | None = None, **kwargs):
        # 1. Extract device early and REMOVE from kwargs to avoid dupe argument error
        device = kwargs.pop("device", "cuda:0")
        
        # 2. Build KinCharModel ON DEVICE immediately
        self._build_kin_char_model(cfg.char_file, device)

        self._motion_lib: Optional[motion_lib.MotionLib] = None
        if cfg.motion_file:
            self._build_motion_lib(cfg.motion_file, cfg, device)

        self._control_mode = cfg.control_mode
        self._key_body_ids: Optional[torch.Tensor] = None

        # Reference buffers
        self._ref_root_pos = None
        self._ref_root_rot = None
        self._ref_root_vel = None
        self._ref_root_ang_vel = None
        self._ref_joint_rot = None
        self._ref_dof_pos = None
        self._ref_dof_vel = None
        self._ref_body_pos = None
        self._ref_contacts = None
        
        # Initialize motion buffers early
        num_envs = cfg.scene.num_envs
        self._motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._motion_times = torch.zeros(num_envs, device=device)

        # 3. Call parent init (creates sim and self.device)
        super().__init__(cfg, device=device, render_mode=render_mode, **kwargs)
        
        # 4. Build buffers
        self._build_data_buffers()

    def _build_kin_char_model(self, char_file: str, device: str) -> None:
        if not char_file:
            raise ValueError("char_file must be specified in configuration")
        _, file_ext = os.path.splitext(char_file)
        if file_ext != ".xml":
            raise ValueError(f"Unsupported character file format: {file_ext}")
        
        # Initialize directly on the target device
        self._kin_char_model = kin_char_model.KinCharModel(device)
        self._kin_char_model.load_char_file(char_file)

    def _build_motion_lib(self, motion_file: str, cfg: Any, device: str) -> None:
        self._motion_lib = motion_lib.MotionLib.from_file(
            motion_file=motion_file,
            char_model=self._kin_char_model,
            device=device,
            contact_info=cfg.use_contact_info,
        )

    def _build_data_buffers(self) -> None:
        num_envs = self.num_envs
        device = self.device
        
        robot_entity: Entity = self.scene.entities["robot"] 
        num_bodies = len(robot_entity.body_names)
        num_dofs = len(robot_entity.joint_names)
        num_joints_kin = self._kin_char_model.get_num_joints()

        self._ref_root_pos = torch.zeros(num_envs, 3, device=device)
        self._ref_root_rot = torch.zeros(num_envs, 4, device=device)
        self._ref_root_rot[:, 3] = 1.0
        self._ref_root_vel = torch.zeros(num_envs, 3, device=device)
        self._ref_root_ang_vel = torch.zeros(num_envs, 3, device=device)
        self._ref_joint_rot = torch.zeros(num_envs, num_joints_kin - 1, 4, device=device)
        self._ref_joint_rot[:, :, 3] = 1.0
        self._ref_dof_pos = torch.zeros(num_envs, num_dofs, device=device)
        self._ref_dof_vel = torch.zeros(num_envs, num_dofs, device=device)
        self._ref_body_pos = torch.zeros(num_envs, num_bodies, 3, device=device)

        if self.cfg.use_contact_info:
            num_contact_bodies = self._kin_char_model.get_num_contact_bodies()
            self._ref_contacts = torch.zeros(num_envs, num_contact_bodies, device=device)

        if self.cfg.key_bodies:
            key_body_ids = []
            for body_name in self.cfg.key_bodies:
                body_id = self._kin_char_model.get_body_id(body_name)
                key_body_ids.append(body_id)
            self._key_body_ids = torch.tensor(key_body_ids, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def robot(self) -> Entity:
        return self.scene.entities["robot"]

    @property
    def char_root_pos(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w

    @property
    def char_root_rot(self) -> torch.Tensor:
        return self.robot.data.root_link_quat_w
    
    @property
    def char_root_vel(self) -> torch.Tensor:
        return self.robot.data.root_link_lin_vel_w

    @property
    def char_root_ang_vel(self) -> torch.Tensor:
        return self.robot.data.root_link_ang_vel_w

    @property
    def char_dof_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def char_dof_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel
    
    @property
    def char_contact_forces(self) -> torch.Tensor:
        return self.robot.data.body_external_force

    # Reference properties (ADDED THESE 2 GETTERS)
    @property
    def ref_root_pos(self) -> torch.Tensor: return self._ref_root_pos
    @property
    def ref_root_rot(self) -> torch.Tensor: return self._ref_root_rot
    @property
    def ref_root_vel(self) -> torch.Tensor: return self._ref_root_vel
    @property
    def ref_root_ang_vel(self) -> torch.Tensor: return self._ref_root_ang_vel
    @property
    def ref_joint_rot(self) -> torch.Tensor: return self._ref_joint_rot
    @property
    def ref_dof_pos(self) -> torch.Tensor: return self._ref_dof_pos
    @property
    def ref_dof_vel(self) -> torch.Tensor: return self._ref_dof_vel
    @property
    def ref_body_pos(self) -> torch.Tensor: return self._ref_body_pos

    # -------------------------------------------------------------------------
    # Motion Logic
    # -------------------------------------------------------------------------
    def sample_motion_state(self, env_ids: torch.Tensor) -> None:
        if self._motion_lib is None: return
        num_sample = len(env_ids)
        motion_ids = self._motion_lib.sample_motions(num_sample)
        self._motion_ids[env_ids] = motion_ids
        motion_times = self._motion_lib.sample_time(motion_ids)
        self._motion_times[env_ids] = motion_times

        frame_data = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        if self.cfg.use_contact_info:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = frame_data
            self._ref_contacts[env_ids] = contacts
        else:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = frame_data

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        self._ref_dof_pos[env_ids] = self._kin_char_model.rot_to_dof(joint_rot)
        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        self._ref_body_pos[env_ids] = body_pos

    def update_motion_time(self, dt: float) -> None:
        if self._motion_lib is None: return
        self._motion_times += dt
        frame_data = self._motion_lib.calc_motion_frame(self._motion_ids, self._motion_times)
        if self.cfg.use_contact_info:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = frame_data
            self._ref_contacts[:] = contacts
        else:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = frame_data

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel
        self._ref_dof_pos[:] = self._kin_char_model.rot_to_dof(joint_rot)
        body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        self._ref_body_pos[:] = body_pos

    def step(self, action: torch.Tensor):
        self.update_motion_time(self.step_dt)
        return super().step(action)

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or len(env_ids) == 0: return
        self.sample_motion_state(env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([
                self._ref_root_pos[env_ids], 
                self._ref_root_rot[env_ids],
                self._ref_root_vel[env_ids],
                self._ref_root_ang_vel[env_ids]
            ], dim=-1),
            env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            self._ref_dof_pos[env_ids], 
            self._ref_dof_vel[env_ids], 
            env_ids=env_ids
        )
        super()._reset_idx(env_ids)