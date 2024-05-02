from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import model
from brax.io import mjcf
from etils import epath
import time
import numpy as np
import jax
from jax import numpy as jp
import mujoco
import mujoco.viewer
import cv2
from tqdm import tqdm
from datetime import datetime

from rl_games.envs.dexmv.relocate import Relocate

def domain_randomize(sys, rng):
  """Randomizes the mjx.Model."""
  @jax.vmap
  def rand(rng):
    # refer to https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=h8mhzKjHQuoL
    #randomness_scale=0.15
    target_position_range=(jp.array([-0.5, -0.5, 0.15]),jp.array([0.5, 0.5, 0.3]))
    target_idx = 2
    
    _, key = jax.random.split(rng, 2)
    
    # qpos, qvel
    #low, hi = -randomness_scale, randomness_scale
    #qpos = sys.init_q + jax.random.uniform(key1, (sys.q_size(),), minval=low, maxval=hi)
    #qvel = jax.random.uniform(key2, (sys.qd_size(),), minval=low, maxval=hi)
    
    # target position
    low, hi = target_position_range
    target_pos = jax.random.uniform(key, (3,), minval=low, maxval=hi)
    body_pos = sys.body_pos.at[target_idx, :].set(target_pos)
    
    #return qpos, qvel, body_pos
    return body_pos,

  body_pos, = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
    "body_pos": 0,
  })

  sys = sys.tree_replace({
    "body_pos": body_pos,
  })

  return sys, in_axes

def save_video(env, rollout):
  def save_video_by_camera(env, rollout, camera=None):
    frames = env.render(rollout, camera=camera)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(1 / env.dt)
    height, width = frames[0].shape[:2]
    
    video = cv2.VideoWriter(f"rollout_{camera}.mp4", fourcc, fps, (width, height))

    for idx in range(len(frames)):
      video.write(frames[idx])
    
    video.release()

  save_video_by_camera(env, rollout, camera=None)
  save_video_by_camera(env, rollout, camera="zeroview")
  save_video_by_camera(env, rollout, camera="sideview")
  save_video_by_camera(env, rollout, camera="frontview")
  save_video_by_camera(env, rollout, camera="backview")

def vis():
  env = Relocate()

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)

  state = jit_env_reset(jax.random.PRNGKey(int(time.time())))
  rollout = [state.pipeline_state]
    
  for i in tqdm(range(900)):
    """
    0 ~ 5: arm translation / rotation
    6 ~ 7: wrist rotation
    8 ~ 29: fingers
    """
    ctrl = np.zeros(env.sys.act_size())
    """
    if i < 300:
      ctrl[:6] = 1.0
    elif i < 600:
      ctrl[:6] = -1.0
    else:
      ctrl[:6] = 0.0
    """
    ctrl = jp.array(ctrl)
    state = jit_env_step(state, ctrl)
    rollout.append(state.pipeline_state)
  
  save_video(env, rollout)

def viewer_test():
  import copy
  
  env = Relocate()
  env1 = Relocate()
  
  model = env.sys.mj_model
  model1 = env.sys.mj_model
  data = mujoco.MjData(model)
  data1 = mujoco.MjData(model1)
  
  with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0
    input("enter when ready")
    while viewer.is_running() and t < 2000:
      t += 1
      
      data.ctrl[1] = 1.0
      mujoco.mj_step(model, data)
      
      data1.qpos = data.qpos
      data1.qvel = data.qvel
      data1.act = data.act
      data1.ctrl = data.ctrl
      data1.xpos = data.xpos
      data1.geom_xpos = data.geom_xpos
      #data1.site_xpos = data.site_xpos
      #data1.cam_xpos = data.cam_xpos
      #data1.light_xpos = data.light_xpos
      #data1.flexvert_xpos = data.flexvert_xpos
      #data1.wrap_xpos = data.wrap_xpos
      
      viewer.sync()
      time.sleep(float(env.sys.dt))
    input("Enter to quit ")

def view():
  env = Relocate()

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)

  state = jit_env_reset(jax.random.PRNGKey(0))
  
  model = env.sys.mj_model
  data = mujoco.MjData(model)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    input("Enter when ready ")
    for i in range(900):
      """
      0 ~ 5: arm translation / rotation
      6 ~ 7: wrist rotation
      8 ~ 29: fingers
      """
      ctrl = np.zeros(env.sys.act_size())
      if i < 300:
        ctrl[1] = 1.0
        ctrl[6:] = 1.0
      elif i < 600:
        ctrl[1] = 0.0
        ctrl[6:] = 0.0
      else:
        ctrl[1] = -0.5
        ctrl[6:] = -0.5
      
      ctrl = jp.array(ctrl)
      state = jit_env_step(state, ctrl)
      
      data.qpos = state.pipeline_state.q
      data.qvel = state.pipeline_state.qd
      data.act = state.pipeline_state.act
      data.ctrl = state.pipeline_state.ctrl
      data.xpos = state.pipeline_state.xpos
      #data.geom_xpos = state.pipeline_state.geom_xpos
      #data.site_xpos = state.pipeline_state.site_xpos
      #data.cam_xpos = state.pipeline_state.cam_xpos
      
      if not viewer.is_running:
        break
      viewer.sync()
    input("Enter to quit ")

#vis()
#viewer_test()
view()

"""
TODO

- [o] 팔 움직이게 joint 추가
  - rotation joint에 대한 damping 조정 (20->10) 하여 불안정성 해결함
- [o] 카메라 위치 조정
- [o] 파이썬 클래스로 환경 조작해보기
  - 손, 물체, target 위치
    - body 이름 목록: sys.link_names (28,)
      ['table', 'target', 'object', 'rh_forearm', 'rh_wrist', 
      'rh_palm', 'rh_ffknuckle', 'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal', 
      'rh_mfknuckle', 'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal', 'rh_rfknuckle', 
      'rh_rfproximal', 'rh_rfmiddle', 'rh_rfdistal', 'rh_lfmetacarpal', 'rh_lfknuckle', 
      'rh_lfproximal', 'rh_lfmiddle', 'rh_lfdistal', 'rh_thbase', 'rh_thproximal', 
      'rh_thhub', 'rh_thmiddle', 'rh_thdistal']
    - body cartesian position 목록: pipeline_state.x.pos (28, 3)
    - 둘이 순서 딱 맞는듯?
- [o] 각 손가락별 조작해보기
  - arm 수직 조작 force 조절로 해결
- [ ] contact 따라하고 파이썬으로 조작해보기
  - body 안에 plastic_collision geom에 이름 할당하기
  - object와 손의 contact 설정하기?
  - state.pipeline_state.contact 안에 link_idx에 array 2개가 있는데 그 안에 손과 물체의 index 있으면 contact인듯?
- [o] action normalize (-1, 1) to (action_min, action_max)
  - mean + action * scale
- [ ] 물체 교체 (다양하게)
- [ ] reward 수정하기
- [ ] 학습시켜보기
"""