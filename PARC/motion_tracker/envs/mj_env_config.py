"""
mjlab Environment Configuration for PARC Motion Tracking.
"""

from __future__ import annotations
import enum
from dataclasses import MISSING, field, dataclass
from typing import Any, List, Tuple

# mjlab imports
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg, 
    RewardTermCfg, SceneEntityCfg, TerminationTermCfg
)
from mjlab.scene import SceneCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.actuator import ActuatorCfg
from mjlab.terrains import TerrainImporterCfg

# Import the standalone functions
import parc.motion_tracker.envs.mj_char_func as mj_funcs

# Enum definitions
class CameraMode(enum.Enum):
    still = 0
    track = 1

class ControlMode(enum.Enum):
    pd = 0
    vel = 1
    torque = 2
    pd_exp = 3
    pd_1d = 4

# Actuator Configs
@dataclass
class PDExpActuatorCfg(ActuatorCfg):
    stiffness: float | dict[str, float] = 100.0
    damping: float | dict[str, float] = 10.0
    effort_limit: float | dict[str, float] = 100.0
    kin_char_model_path: str = ""

    def build(self, entity, target_ids, target_names):
        from parc.motion_tracker.envs.mj_char_env import PDExpActuator
        return PDExpActuator(self, entity, target_ids, target_names)

@dataclass
class PD1DActuatorCfg(ActuatorCfg):
    stiffness: float = 100.0
    damping: float = 10.0
    effort_limit: float = 100.0

    def build(self, entity, target_ids, target_names):
        from parc.motion_tracker.envs.mj_char_env import PD1DActuator
        return PD1DActuator(self, entity, target_ids, target_names)

# Scene Config
@dataclass
class MJCharSceneCfg(SceneCfg):
    num_envs: int = 4096
    env_spacing: float = 5.0
    terrain: TerrainImporterCfg = field(default_factory=lambda: TerrainImporterCfg(
        terrain_type="plane",
        env_spacing=5.0
    ))
    robot: EntityCfg = field(default_factory=EntityCfg)

    def __post_init__(self):
        if "robot" not in self.entities:
            self.entities["robot"] = self.robot

# Main Config
@dataclass
class MJCharEnvCfg(ManagerBasedRlEnvCfg):
    decimation: int = 6
    scene: MJCharSceneCfg = field(default_factory=MJCharSceneCfg)
    
    # Pass actual function objects here!
    observations: dict[str, ObservationGroupCfg] = field(default_factory=lambda: {
        "policy": ObservationGroupCfg(
            terms={
                "root_height": ObservationTermCfg(func=mj_funcs.root_height),
                "root_rot": ObservationTermCfg(func=mj_funcs.root_rotation_tan_norm, params={"global_obs": False}),
                "root_lin_vel": ObservationTermCfg(func=mj_funcs.root_lin_vel, params={"global_obs": False}),
                "root_ang_vel": ObservationTermCfg(func=mj_funcs.root_ang_vel, params={"global_obs": False}),
                "joint_rot": ObservationTermCfg(func=mj_funcs.joint_rotation_tan_norm),
                "joint_vel": ObservationTermCfg(func=mj_funcs.joint_vel),
                "key_body_pos": ObservationTermCfg(func=mj_funcs.key_body_positions, params={"key_bodies": [], "global_obs": False}),
            },
            enable_corruption=False,
            concatenate_terms=True
        ),
        "target": ObservationGroupCfg(
            terms={
                "tar_root_pos": ObservationTermCfg(func=mj_funcs.target_root_pos, params={"tar_obs_steps": [1]}),
                "tar_root_rot": ObservationTermCfg(func=mj_funcs.target_root_rot, params={"tar_obs_steps": [1]}),
                "tar_joint_rot": ObservationTermCfg(func=mj_funcs.target_joint_rot, params={"tar_obs_steps": [1]}),
                "tar_key_body_pos": ObservationTermCfg(func=mj_funcs.target_key_body_pos, params={"tar_obs_steps": [1], "key_bodies": []}),
            },
            enable_corruption=False,
            concatenate_terms=True
        )
    })

    rewards: dict[str, RewardTermCfg] = field(default_factory=lambda: {
        "pose": RewardTermCfg(func=mj_funcs.pose_reward, weight=0.5, params={"scale": 2.0}),
        "vel": RewardTermCfg(func=mj_funcs.vel_reward, weight=0.1, params={"scale": 0.1}),
        "root_pos": RewardTermCfg(func=mj_funcs.root_pos_reward, weight=0.15, params={"scale": 2.0, "track_root_h": True}),
        "root_vel": RewardTermCfg(func=mj_funcs.root_vel_reward, weight=0.1, params={"scale": 2.0}),
        "key_pos": RewardTermCfg(func=mj_funcs.key_pos_reward, weight=0.15, params={"scale": 10.0, "key_bodies": []}),
    })

    terminations: dict[str, TerminationTermCfg] = field(default_factory=lambda: {
        "time_out": TerminationTermCfg(func=mj_funcs.time_out, time_out=True),
        "root_height": TerminationTermCfg(func=mj_funcs.root_height_termination, time_out=False, params={"termination_height": 0.3}),
    })

    actions: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    
    char_file: str = ""
    control_mode: ControlMode = ControlMode.pd_exp
    episode_length_s: float = 10.0
    motion_file: str = ""
    use_contact_info: bool = False
    key_bodies: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.sim.dt = 1.0 / 60.0
        
        if "policy" in self.observations and "key_body_pos" in self.observations["policy"].terms:
            self.observations["policy"].terms["key_body_pos"].params["key_bodies"] = self.key_bodies
            
        if "target" in self.observations and "tar_key_body_pos" in self.observations["target"].terms:
            self.observations["target"].terms["tar_key_body_pos"].params["key_bodies"] = self.key_bodies

        if "key_pos" in self.rewards:
            self.rewards["key_pos"].params["key_bodies"] = self.key_bodies