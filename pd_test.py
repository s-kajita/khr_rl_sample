import os
import pickle
import shutil
import numpy as np

import genesis as gs

from collections import deque     # dequeはリングバッファ
MAX_LOG_SIZE = 500
simT = deque([],maxlen=MAX_LOG_SIZE)
dofs_log = deque([],maxlen=MAX_LOG_SIZE)
dofs_force_log = deque([],maxlen=MAX_LOG_SIZE)

import matplotlib.pyplot as plt
import numpy as np


env_cfg = []


def get_cfgs():
    env_cfg = {
        "num_actions": 12, #強化学習する関節の数
       # PD
        "kp": np.array([25]*12),
        "kd": np.array([0.5]*12),
        #"armature": np.array([0.149,0.401,0.434,0.536,0.226,0.070]*2),
        # termination
        "termination_if_roll_greater_than": 15,  # degree
        "termination_if_pitch_greater_than": 15,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.27],      #本体の初期高さ
        "base_init_quat": [1.0, 0.0, 0.0, 0.0], #本体の初期姿勢
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,

        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }

    return env_cfg


Device="cuda"

gs.init(logging_level="warning")

env_cfg = get_cfgs()
Dtime = 0.01

# create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=Dtime,
    ),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=int(0.5 / Dtime),
        camera_pos=(1.0, 0.0, 0.5),
        camera_lookat=(0.0, 0.0, 0.2),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    rigid_options=gs.options.RigidOptions(
        dt=Dtime,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,
)

# add plain
plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# add robot
base_init_pos = env_cfg["base_init_pos"]
base_init_quat = env_cfg["base_init_quat"]
KHR_roter_inertia = [0.01]*22
robot = scene.add_entity(
    gs.morphs.URDF(
        file="../assets/khr3hv/urdf/khr3hv_12dof.urdf",  #頭の位置が正しく表示される
        pos=base_init_pos,
        quat=base_init_quat,
    ),
    #dofs_armature=np.array([0.01]*22),
)

# build
scene.build()

jnt_names = [
    "l_hip_yaw", #default_joint_anglesの名前に合わせて変更
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
]
dofs_idx = [robot.get_joint(name).dofs_idx_local[0] for name in jnt_names]

############ オプション：制御ゲインの設定 ############
# 位置ゲインの設定
robot.set_dofs_kp(
    kp             = env_cfg["kp"],
    dofs_idx_local = dofs_idx,
)
# 速度ゲインの設定
robot.set_dofs_kv(
    kv             = env_cfg["kd"],
    dofs_idx_local = dofs_idx,
)
'''
# 安全のための力の範囲設定
robot.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)
'''

# 初期姿勢設定
default_joint_angles = np.array([0,0,-0.2,0.4,-0.2,0]*2)
 
# PD制御
for i in range(300):
    robot.control_dofs_position(default_joint_angles, dofs_idx)
    scene.step()
    
    simT.append(i*Dtime)
    # 制御に使用する関節
    dofs_log.append(robot.get_dofs_position(dofs_idx).tolist())
    dofs_force_log.append(robot.get_dofs_control_force(dofs_idx).tolist()) 

print("r_foot: ",robot.get_link('r_foot').get_pos())
print("base_link: ",robot.get_link('base_link').get_pos())
print("robot.get_pos():",robot.get_pos())

#================ show graph =======================
plt.figure()
plt.subplot(211)
plt.plot(simT,np.rad2deg(dofs_log))
plt.ylabel('[deg]')
plt.title(os.path.basename(__file__))

plt.subplot(212)
plt.plot(simT,dofs_force_log)
plt.ylabel('[Nm]')
plt.xlabel('time [s]')



#================ finish =======================
plt.show(block=False)
input('Hit enter to close figs')
plt.close("all")


