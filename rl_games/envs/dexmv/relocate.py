from brax import base
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import model
from brax.io import mjcf
import numpy as np
import jax
from jax import numpy as jp
import mujoco

#jax.config.update("jax_debug_nans", True)

"""
TODO
- [ ] std nan 문제 해결하기
- [ ] 공 이외의 object 지원하기
- [ ] qpos, qvel randomization
- [ ] target pose randomization
"""

class Relocate(PipelineEnv):
  def __init__(
      self,
      object_name="mug",
      object_scale=1,
      randomness_scale=0.0,
      target_position_low=(-0.5, -0.5, 0.15),
      target_position_hi=(0.5, 0.5, 0.3),
      backend="mjx",
      **kwargs,
  ):
    assert backend == "mjx"

    # load model
    path = "./rl_games/envs/assets/shadow_hand/table.xml"
    mjmodel = mujoco.MjModel.from_xml_path(path)
    sys = mjcf.load_model(mjmodel)
    sys = sys.tree_replace({
      'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
      'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
      'opt.iterations': 1,
      'opt.ls_iterations': 4,
    })

    # kwargs
    kwargs["n_frames"] = kwargs.get("n_frames", 5)

    # super
    super().__init__(sys=sys, backend=backend, **kwargs)

    # get options
    self.object_name = object_name
    self.object_scale = object_scale
    self.randomness_scale = randomness_scale
    self.target_position_low = target_position_low
    self.target_position_hi = target_position_hi
    
    # setup action range (for normalization)
    self.action_mean = jp.mean(self.sys.actuator_ctrlrange, axis=1)
    self.action_scale = 0.5 * (self.sys.actuator_ctrlrange[:, 1] - self.sys.actuator_ctrlrange[:, 0])
    print(f"Action size {self.sys.act_size()}")
    print(f"Action mean {self.action_mean}")
    print(f"Action sacle {self.action_scale}")
    
    # get palm, object, target index
    self.palm_body_idx = self.sys.link_names.index("rh_palm")
    self.object_body_idx = self.sys.link_names.index("object")
    self.target_body_idx = self.sys.link_names.index("target")

  def reset(self, rng: jax.Array) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self.randomness_scale, self.randomness_scale
    qpos = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
    qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)
    
    self.target_pos = np.random.uniform(low=self.target_position_low, high=self.target_position_hi)
    self.sys.mj_model.body_pos[self.target_body_idx + 1, :] = self.target_pos
    
    print(f"Init target position {self.target_pos}")

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state, self.action_mean)
    
    reward, done, zero = jp.zeros(3)
    metrics = {
        "task_reward": zero,
        "quadctrl_reward": zero,
        "healthy_reward": zero,
        "object_x_pos": zero,
        "object_y_pos": zero,
        "object_z_pos": zero,
        "target_x_pos": zero,
        "target_y_pos": zero,
        "target_z_pos": zero,
        "dist_object_target": zero,
        "dist_object_palm": zero,
        "dist_palm_target": zero,
        "success_reward": zero
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    # get action (input action is between -1.0 and 1.0)
    scaled_action = self.action_mean + self.action_scale * action
    
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, scaled_action)
    
    # calculate get positions
    palm_pos = pipeline_state.xpos[self.palm_body_idx + 1]
    object_pos = pipeline_state.xpos[self.object_body_idx + 1]
    target_pos = self.target_pos
    
    # calculate distance between bodies
    distance_palm_object = jp.linalg.norm(palm_pos - object_pos)
    distance_object_target = jp.linalg.norm(object_pos - target_pos)
    distance_palm_target = jp.linalg.norm(palm_pos - target_pos)
    
    """
    # Case: original dexmv reward
    # calculate task reward
    task_reward = -0.1 * distance_palm_object # hand - object distance
    
    # check contact (palm and object)
    contact_idx = jp.reshape(jp.concatenate(pipeline_state.contact.link_idx), (2, -1))
    is_contact = jp.where(contact_idx == jp.array([[self.palm_body_idx], [self.object_body_idx]]), 1.0, 0.0)
    is_contact = jp.where(contact_idx == jp.array([[self.object_body_idx], [self.palm_body_idx]]), 1.0, is_contact)
    is_contact = jp.any(is_contact)
    
    contact_reward = 0.1
    contact_reward += jp.where(object_pos[2] > 0.04, 50 * (object_pos[2] - 0.04), 0.0)
    contact_reward += jp.where(object_pos[2] > 0.055, 2.0 - 0.5 * distance_palm_target - 1.5 * distance_object_target, 0.0)
    contact_reward += jp.where(distance_object_target < 0.1, 1 / (distance_object_target + 0.01), 0.0)
    
    task_reward += jp.where(is_contact, contact_reward, 0.0)
    """

    # Case: lift the object as high as possible
    task_reward = -0.1 * distance_palm_object
    task_reward += jp.where(object_pos[2] > 0.04, 50 * (object_pos[2] - 0.04), 0.0)
    
    """
    # Case: just move palm to the target position
    task_reward = -0.5 * distance_palm_target
    """

    # healthy reward: object on the table
    is_healthy = jp.where(-0.6 <= object_pos[0], 1.0, 0.0)
    is_healthy = jp.where(object_pos[0] <= 0.6, is_healthy, 0.0)
    is_healthy = jp.where(-0.6 <= object_pos[1], is_healthy, 0.0)
    is_healthy = jp.where(object_pos[1] <= 0.6, is_healthy, 0.0)
    is_healthy = jp.where(0.0 <= object_pos[2], is_healthy, 0.0)
    
    # control cost
    ctrl_cost = 0.1 * jp.sum(jp.square(action))
    
    # success: object - target distance < 0.1 * object size
    success = jp.where(distance_object_target <= 0.1 * 0.04 * 2, 1.0, 0.0)

    obs = self._get_obs(pipeline_state, scaled_action)
    
    #reward = task_reward + is_healthy - ctrl_cost
    reward = task_reward # original dexmv env only considers task reward
    
    done = 0.0
    state.metrics.update(
        task_reward=task_reward,
        quadctrl_reward=-ctrl_cost,
        healthy_reward=is_healthy,
        object_x_pos=object_pos[0],
        object_y_pos=object_pos[1],
        object_z_pos=object_pos[2],
        target_x_pos=target_pos[0],
        target_y_pos=target_pos[1],
        target_z_pos=target_pos[2],
        dist_object_target=distance_object_target,
        dist_object_palm=distance_palm_object,
        dist_palm_target=distance_palm_target,
        success_reward=success
    )

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
    qpos = pipeline_state.qpos
    
    palm_pos = pipeline_state.xpos[self.palm_body_idx + 1]
    object_pos = pipeline_state.xpos[self.object_body_idx + 1]
    target_pos = self.target_pos

    # external_contact_forces are excluded
    return jp.concatenate([
        qpos,
        palm_pos - object_pos,
        palm_pos - target_pos,
        object_pos - target_pos
    ])
