"""
Launch script for PARC mjlab environment.
"""
import argparse
import os
import torch
import gymnasium as gym
import mujoco

from parc.motion_tracker.envs.mj_char_env import MJCharEnv
from parc.motion_tracker.envs.mj_env_config import MJCharEnvCfg, PDExpActuatorCfg, ControlMode
from mjlab.entity import EntityArticulationInfoCfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char-file", type=str, required=True)
    parser.add_argument("--motion-file", type=str, default="")
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()

    cfg = MJCharEnvCfg()
    cfg.char_file = os.path.abspath(args.char_file)
    cfg.motion_file = os.path.abspath(args.motion_file) if args.motion_file else ""
    cfg.scene.num_envs = args.num_envs
    cfg.key_bodies = ["right_hand", "left_hand", "right_foot", "left_foot"]

    # --- Configure Scene Spec (Memory Limits) ---
    def increase_scene_limits(spec: mujoco.MjSpec):
        # Increase limits to prevent 'nefc overflow'
        spec.njmax = 2000
        spec.nconmax = 1000
        
    cfg.scene.spec_fn = increase_scene_limits

    # --- Configure Robot Entity ---
    char_path = cfg.char_file
    
    def load_robot_spec():
        spec = mujoco.MjSpec.from_file(char_path)
        
        # 1. Remove existing actuators
        for actuator in list(spec.actuators):
            spec.delete(actuator)
            
        # 2. Remove floor/plane geometries (mjlab handles terrain)
        if spec.worldbody:
            for geom in list(spec.worldbody.geoms):
                if geom.type == mujoco.mjtGeom.mjGEOM_PLANE or (geom.name and "floor" in geom.name):
                    spec.delete(geom)
            
            # 3. Remove lights (mjlab scene has its own lighting)
            for light in list(spec.worldbody.lights):
                spec.delete(light)

        return spec

    cfg.scene.robot.spec_fn = load_robot_spec
    
    # --- Configure Actuators ---
    if cfg.control_mode == ControlMode.pd_exp:
        # Create actuator config
        pd_actuator = PDExpActuatorCfg(
            target_names_expr=(".*",), # Regex for all joints
            stiffness=500.0,
            damping=50.0,
            effort_limit=1000.0,
            kin_char_model_path=cfg.char_file
        )
        # Assign to entity articulation
        cfg.scene.robot.articulation = EntityArticulationInfoCfg(
            actuators=(pd_actuator,)
        )

    # --- Initialize Environment ---
    print(f"Launching environment with {cfg.scene.num_envs} envs on CUDA...")
    env = MJCharEnv(cfg=cfg, render_mode="human", device="cuda:0")
    
    print("Resetting environment...")
    obs, _ = env.reset()
    
    print(f"Observation shape (Policy): {obs['policy'].shape}")
    if 'target' in obs:
        print(f"Observation shape (Target): {obs['target'].shape}")

    print("Starting simulation loop...")
    while True:
        action_dim = env.action_space.shape[1]
        actions = 0.01 * torch.randn(env.num_envs, action_dim, device=env.device)
        env.step(actions)

if __name__ == "__main__":
    main()