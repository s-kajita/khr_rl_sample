import argparse
import os
import pickle
from importlib import metadata
import pandas as pd

import torch
import numpy as np
import csv
import math

import pygame
import time

from scipy.signal import find_peaks

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from khrbase_env import KHREnv
import matplotlib.pyplot as plt

env = []

pygame.init()
pygame.joystick.init()
# ジョイスティックの取得
if pygame.joystick.get_count() == 0:
    print("ジョイスティックが見つかりません")
    exit()
joy = pygame.joystick.Joystick(0)
joy.init()
print("使用デバイス:", joy.get_name())


# main()    
# 状態保持
prev_btnA = False
prev_btnB = False
prev_btnX = False
prev_btnY = False
prev_btnR2 = False
max_vel = 0.2
min_vel = -0.2
max_ang_vel = 0.5
min_ang_vel = -0.5
cmd_lin_vel_x = 0.0
cmd_lin_vel_y = 0.0
cmd_ang_vel = 0.0


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="khrbase")
parser.add_argument("-I","--ckpt", type=int, default=10000)
parser.add_argument('--eval', type=int, default=1)
args = parser.parse_args()

gs.init()

log_dir = f"logs/{args.exp_name}"
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
reward_cfg["reward_scales"] = {}

command_cfg = {
    "num_commands": 3,
    "heading_command":False,
    "lin_vel_x_range":[cmd_lin_vel_x, cmd_lin_vel_x],  #0.2が最大値
    "lin_vel_y_range": [cmd_lin_vel_y, cmd_lin_vel_x], #0.2が最大値
    "ang_vel_range": [cmd_ang_vel, cmd_ang_vel],#0.5が最大値
    #"heading": [-3.14, 3.14],
}
env_cfg["episode_length_s"] = 100
'''
env_cfg["base_init_pos"] = [0, 0, 0.35]
env_cfg['randomize_friction'] = False
env_cfg['randomize_mass'] = False
env_cfg['randomize_com'] = False
env_cfg['randomize_kp'] = False
env_cfg['randomize_kd'] = False
obs_cfg['randomize_armature'] = False
obs_cfg['add_noise'] = True
env_cfg['height_scan'] = True
obs_cfg['obs_history'] = False
obs_cfg['frame_stack'] = 5
command_cfg['heading_command'] = True
'''

env = KHREnv(
    num_envs=1,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
    show_viewer=True,
)

runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
runner.load(resume_path)
policy = runner.get_inference_policy(device=gs.device)

time = 0
time_list = []


obs = env.reset()
with torch.no_grad():
    while True:

        pygame.event.pump()
        # アナログ入力
        x = joy.get_axis(1) # 左右
        y = joy.get_axis(0) # 上下
        z = joy.get_axis(3) #回転方向

        btnA = joy.get_button(0) == 1 # A PScontroller ☓
        btnB = joy.get_button(1) == 1 # B
        btnX = joy.get_button(2) == 1 # X
        btnY = joy.get_button(3) == 1 # Y
        btnR2 = joy.get_axis(5) > 0.5 # C
        
        cmd_lin_vel_x   = np.clip(-x*max_vel*1.1, min_vel, max_vel)
        cmd_lin_vel_y   = np.clip(-y*max_vel*1.1, min_vel, max_vel)
        cmd_ang_vel = np.clip(-z*max_ang_vel*1.1, min_ang_vel, max_ang_vel)
        cmd_lin_vel_x = round(cmd_lin_vel_x,3)
        cmd_lin_vel_y = round(cmd_lin_vel_y,3)
        cmd_ang_vel = round(cmd_ang_vel, 3)
        #print(btnA)
        
        if btnA:
            env.push_robot()
        

        env.commands[0, 0] = cmd_lin_vel_x
        env.commands[0, 1] = cmd_lin_vel_y
        env.commands[0, 2] = cmd_ang_vel
        #env.commands[0, 3] = cmd_ang_vel

        actions = policy(obs)
        obs, rews, dones, infos = env.step(actions)

        dt = 0.02
        time = time + 1
        time_list.append(time / 50) 

        
        #if dones[0] or time > 200:
        #    break
        