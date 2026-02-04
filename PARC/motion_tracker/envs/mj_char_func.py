"""
Standalone observation, reward, and termination functions for MJCharEnv.
"""
from __future__ import annotations
from typing import Any, List
import torch
import parc.util.torch_util as torch_util

# =============================================================================
# Observation Functions
# =============================================================================

def root_height(env: Any) -> torch.Tensor:
    return env.char_root_pos[:, 2:3]

def root_rotation_tan_norm(env: Any, global_obs: bool = False) -> torch.Tensor:
    root_rot = env.char_root_rot
    if global_obs:
        return torch_util.quat_to_tan_norm(root_rot)
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    return torch_util.quat_to_tan_norm(torch_util.quat_mul(heading_inv_rot, root_rot))

def root_lin_vel(env: Any, global_obs: bool = False) -> torch.Tensor:
    if global_obs: return env.char_root_vel
    heading_inv_rot = torch_util.calc_heading_quat_inv(env.char_root_rot)
    return torch_util.quat_rotate(heading_inv_rot, env.char_root_vel)

def root_ang_vel(env: Any, global_obs: bool = False) -> torch.Tensor:
    if global_obs: return env.char_root_ang_vel
    heading_inv_rot = torch_util.calc_heading_quat_inv(env.char_root_rot)
    return torch_util.quat_rotate(heading_inv_rot, env.char_root_ang_vel)

def joint_rotation_tan_norm(env: Any) -> torch.Tensor:
    joint_rot = env._kin_char_model.dof_to_rot(env.char_dof_pos)
    flat_rot_obs = torch_util.quat_to_tan_norm(joint_rot.view(-1, 4))
    return flat_rot_obs.view(env.num_envs, -1)

def joint_vel(env: Any) -> torch.Tensor:
    return env.char_dof_vel

def key_body_positions(env: Any, key_bodies: List[str], global_obs: bool = False) -> torch.Tensor:
    if not key_bodies: return torch.zeros(env.num_envs, 0, device=env.device)
    key_body_ids = [env._kin_char_model.get_body_id(name) for name in key_bodies]
    joint_rot = env._kin_char_model.dof_to_rot(env.char_dof_pos)
    body_pos, _ = env._kin_char_model.forward_kinematics(env.char_root_pos, env.char_root_rot, joint_rot)
    key_pos = body_pos[:, key_body_ids, :] - env.char_root_pos.unsqueeze(1)
    
    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(env.char_root_rot).unsqueeze(1).expand(-1, len(key_body_ids), -1)
        flat_local_pos = torch_util.quat_rotate(heading_inv_rot.reshape(-1, 4), key_pos.reshape(-1, 3))
        key_pos = flat_local_pos.view(env.num_envs, len(key_body_ids), 3)
    return key_pos.view(env.num_envs, -1)

# Target Observations
def target_root_pos(env: Any, tar_obs_steps: List[int] = [1]) -> torch.Tensor:
    if env._motion_lib is None: return torch.zeros(env.num_envs, 0, device=env.device)
    results = []
    for step in tar_obs_steps:
        future_time = env._motion_times + step * env.step_dt
        frame = env._motion_lib.calc_motion_frame(env._motion_ids, future_time)
        results.append(frame[0]) # root_pos
    return torch.cat(results, dim=-1)

def target_root_rot(env: Any, tar_obs_steps: List[int] = [1]) -> torch.Tensor:
    if env._motion_lib is None: return torch.zeros(env.num_envs, 0, device=env.device)
    results = []
    for step in tar_obs_steps:
        future_time = env._motion_times + step * env.step_dt
        frame = env._motion_lib.calc_motion_frame(env._motion_ids, future_time)
        results.append(torch_util.quat_to_tan_norm(frame[1])) # root_rot
    return torch.cat(results, dim=-1)

def target_joint_rot(env: Any, tar_obs_steps: List[int] = [1]) -> torch.Tensor:
    if env._motion_lib is None: return torch.zeros(env.num_envs, 0, device=env.device)
    results = []
    for step in tar_obs_steps:
        future_time = env._motion_times + step * env.step_dt
        frame = env._motion_lib.calc_motion_frame(env._motion_ids, future_time)
        joint_rot = frame[4].view(env.num_envs, -1, 4)
        flat_rot = torch_util.quat_to_tan_norm(joint_rot.view(-1, 4))
        results.append(flat_rot.view(env.num_envs, -1))
    return torch.cat(results, dim=-1)

def target_key_body_pos(env: Any, tar_obs_steps: List[int] = [1], key_bodies: List[str] = []) -> torch.Tensor:
    if not key_bodies or env._motion_lib is None: return torch.zeros(env.num_envs, 0, device=env.device)
    key_ids = [env._kin_char_model.get_body_id(n) for n in key_bodies]
    results = []
    for step in tar_obs_steps:
        future_time = env._motion_times + step * env.step_dt
        root_pos, root_rot, _, _, joint_rot, _ = env._motion_lib.calc_motion_frame(env._motion_ids, future_time)[:6]
        body_pos, _ = env._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        key_pos = body_pos[:, key_ids, :] - root_pos.unsqueeze(1)
        results.append(key_pos.view(env.num_envs, -1))
    return torch.cat(results, dim=-1)

def target_contacts(env: Any, tar_obs_steps: List[int] = [1]) -> torch.Tensor:
    if not env.cfg.use_contact_info: return torch.zeros(env.num_envs, 0, device=env.device)
    results = []
    for step in tar_obs_steps:
        future_time = env._motion_times + step * env.step_dt
        frame = env._motion_lib.calc_motion_frame(env._motion_ids, future_time)
        results.append(frame[6])
    return torch.cat(results, dim=-1)

def character_contacts(env: Any, contact_threshold: float = 1e-5) -> torch.Tensor:
    try:
        contact_forces = env.char_contact_forces
        contact_ids = env._kin_char_model.get_contact_body_ids()
        forces = contact_forces[:, contact_ids, :]
        return (torch.norm(forces, dim=-1) > contact_threshold).float()
    except Exception:
        # Fallback if contacts not available
        return torch.zeros(env.num_envs, 0, device=env.device)

# =============================================================================
# Reward Functions
# =============================================================================

def pose_reward(env: Any, scale: float = 2.0) -> torch.Tensor:
    curr_rot = env._kin_char_model.dof_to_rot(env.char_dof_pos)
    diff_angle = torch_util.quat_to_exp_map(torch_util.quat_diff(curr_rot, env.ref_joint_rot))
    return torch.exp(-scale * torch.mean(torch.sum(diff_angle ** 2, dim=-1), dim=-1))

def vel_reward(env: Any, scale: float = 0.1) -> torch.Tensor:
    return torch.exp(-scale * torch.sum((env.char_dof_vel - env.ref_dof_vel)**2, dim=-1))

def root_pos_reward(env: Any, scale: float = 2.0, track_root_h: bool = True) -> torch.Tensor:
    diff = env.char_root_pos - env.ref_root_pos
    if not track_root_h: diff = diff[:, :2]
    return torch.exp(-scale * torch.sum(diff**2, dim=-1))

def root_vel_reward(env: Any, scale: float = 2.0) -> torch.Tensor:
    return torch.exp(-scale * torch.sum((env.char_root_vel - env.ref_root_vel)**2, dim=-1))

def key_pos_reward(env: Any, scale: float = 10.0, key_bodies: List[str] = None) -> torch.Tensor:
    if not key_bodies or env._key_body_ids is None: return torch.ones(env.num_envs, device=env.device)
    joint_rot = env._kin_char_model.dof_to_rot(env.char_dof_pos)
    body_pos, _ = env._kin_char_model.forward_kinematics(env.char_root_pos, env.char_root_rot, joint_rot)
    diff = body_pos[:, env._key_body_ids, :] - env.ref_body_pos[:, env._key_body_ids, :]
    return torch.exp(-scale * torch.mean(torch.sum(diff**2, dim=-1), dim=-1))

# =============================================================================
# Termination Functions
# =============================================================================

def time_out(env: Any) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

def root_height_termination(env: Any, termination_height: float = 0.3) -> torch.Tensor:
    return env.char_root_pos[:, 2] < termination_height