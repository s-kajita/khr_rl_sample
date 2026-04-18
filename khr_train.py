import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from khr_env import KHREnv


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            #"hidden_dims": [512, 256, 128],
            "hidden_dims": [128, 64, 32],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            #"hidden_dims": [512, 256, 128],
            "hidden_dims": [128, 64, 32],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy","privileged"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "l_hip_yaw": 0.0,
            "l_hip_roll": 0.0,
            "l_hip_pitch": -0.2,
            "l_knee_pitch": 0.4,
            "l_ankle_pitch": -0.2,
            "l_ankle_roll": 0.0,

            "r_hip_yaw": 0.0,
            "r_hip_roll": 0.0,
            "r_hip_pitch": -0.2,
            "r_knee_pitch": 0.4,
            "r_ankle_pitch": -0.2,
            "r_ankle_roll": 0.0,
        },
        "joint_names": [
            "l_hip_yaw",
            "l_hip_roll",
            "l_hip_pitch",
            "l_knee_pitch",
            "l_ankle_pitch",
            "l_ankle_roll",

            "r_hip_yaw",
            "r_hip_roll",
            "r_hip_pitch",
            "r_knee_pitch",
            "r_ankle_pitch",
            "r_ankle_roll",
        ],

        # PD
        "kp": 25.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 50,  # degree
        "termination_if_pitch_greater_than": 50,
        "termination_if_ankle_distance_smaller_than":0.085,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.28],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    
        #domain randomization
        'randomize_friction': True,
        'friction_range': [0.1, 1.5],
        'randomize_base_mass': True,
        'mass_range': [-0.1,0.5],
        'randomize_com': True,
        'com_range': [-0.02, 0.02],
        'randomize_kp': False,
        'kp_scale_range': [0.9, 1.1],
        'randomize_kd' : False,
        'kd_scale_range': [0.8, 1.2],

        'push_interval_s': 5,
        'Mode_push_vel': True,
        'Mode_push_power': False,
        'max_push_vel_xy': 0.2,#m/s
        'max_push_force': 20, #N

    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            
        },

        'add_noise': True,
        "obs_noise": {
            'ang_vel': 0.1,
            "gravity": 0.05,
            "dof_pos": 0.05,
            "dof_vel": 0.1, 
            "action" : 0.0,
        }
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.254,
        "feet_height_target": 0.03,
        "reward_scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 1.0,
            
            "orientation":-5.0,
            "lin_vel_z": -0.1,
            "ang_vel_xy": -0.2,
            "base_height": -10.0,
            "gait_contact" : 0.18,
            "gait_swing": -0.05,
            "contact_no_vel": -0.2,
            #"feet_swing_height": -30.0,
            "feet_clearance": 0.2,
            #feet_distance" : 0.2,
            "hip_pos": -1.0,
            "alive" : 0.5,

            "action_rate": -0.05, #-0.05
            #action_smoothness": -0.001,
            "similar_to_default": -0.1,
            "dof_vel": -0.001,
            "acceleration" : -0.0001,
            "joint_torques":-0.0005,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.2, 0.2],
        "lin_vel_y_range": [-0.2, 0.2],
        "ang_vel_range": [-0.5, 0.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="khr-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = KHREnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
