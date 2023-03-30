import gym
import tqdm
import numpy as np
import omnigibson as og
import matplotlib.pyplot as plt

from time import time
from omnigibson import object_states
from omnigibson.objects import PrimitiveObject
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.asset_utils import get_available_og_scenes
from omnigibson.envs.env_base import Environment
from omnigibson.macros import gm

PROFILING_FIELDS = ["total time", "action time", "physics time", "render time", "non physics time", "get observation time", "task time"]

class ProfilingEnv(Environment):
    def step(self, action):
        start = time()
        # If the action is not a dictionary, convert into a dictionary
        if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
            action_dict = dict()
            idx = 0
            for robot in self.robots:
                action_dim = robot.action_dim
                action_dict[robot.name] = action[idx: idx + action_dim]
                idx += action_dim
        else:
            # Our inputted action is the action dictionary
            action_dict = action

        # Iterate over all robots and apply actions
        for robot in self.robots:
            robot.apply_action(action_dict[robot.name])
        action_end = time()
        # Run simulation step
        # Possibly force playing
        for i in range(og.sim.n_physics_timesteps_per_render):
            super(type(og.sim), og.sim).step(render=False)
        physics_end = time()
        og.sim.render()
        render_end = time()
        # Additionally run non physics things if we have a valid scene
        if og.sim._scene is not None:
            og.sim._omni_update_step()
            if og.sim.is_playing():
                og.sim._non_physics_step()
                if gm.ENABLE_TRANSITION_RULES:
                    og.sim._transition_rule_step()
        non_physics_end = time()
        # Grab observations
        obs = self.get_obs()
        obs_end = time()
        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)

        if done and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Increment step
        self._current_step += 1
        end = time()
        ret = [end-start, action_end-start, physics_end-action_end, render_end-physics_end, \
            non_physics_end-render_end, obs_end-non_physics_end, end-obs_end]
        if self._current_step % 100 == 0:
            print("total time: {:.3f} ms, action time: {:.3f} ms, physics time: {:.3f} ms, render time: {:.3f} ms, non physics time: {:.3f} ms, get obs time: {:.3f} ms, task time: {:.3f} ms".format(*ret))
        return obs, reward, done, info, ret


def plot_results(results_dict, output_fp):
    num_plots = np.sum([len(results) for results in results_dict.values()])
    cur_plot = 1
    plt.figure(figsize=(7, 3.5 * num_plots))
    for cat, results in results_dict.items():
        n_candidates = len(list(results.values())[0])
        for field in results:
            result = [results[field][candidate][0] for candidate in results[field]]
            ax = plt.subplot(num_plots, 1, cur_plot)
            ax.set_xlabel(f"{cat}: {field}")
            ax.set_ylabel("time (ms)")
            plt.bar(range(n_candidates), result, tick_label=list(results[field].keys()))
            cur_plot += 1
    plt.tight_layout()
    plt.savefig(output_fp)


def benchmark_objects(env):
    NUM_ITER = 30
    NUM_OBJECTS_PER_ITER = 10
    NUM_STEP_PER_ITER = 200

    og.sim.stop()
    env.reload({"scene": {"type": "Scene"}})
    env.reset()

    results = {field: {} for field in PROFILING_FIELDS}
    for i in tqdm.trange(NUM_ITER):
        objs = []
        cur_results = []
        for j in range(NUM_OBJECTS_PER_ITER):
            obj = PrimitiveObject(
                prim_path=f"/World/obj_{i}_{j}",
                name=f"obj_{i}_{j}",
                primitive_type="Cube",
            )
            og.sim.import_object(obj)
            obj.set_position(np.array([i, j, 1]) * 1.2)
        objs.append(obj)
        og.sim.step()   # always taks a step after importing objects
        for _ in range(NUM_STEP_PER_ITER):
            cur_result = env.step(None)
            cur_results.append(cur_result[4])
        cur_results = np.array(cur_results)
        for k, field in enumerate(PROFILING_FIELDS):
            results[field][(i + 1) * NUM_OBJECTS_PER_ITER] = [
                np.mean(cur_results[-100:, k]), np.std(cur_results[-100:, k]), np.median(cur_results[-100:, k]), np.max(cur_results[-100:, k]), np.min(cur_results[-100:, k])
            ]
        for obj in objs:
            obj.sleep()

    return results



def benchmark_robots(env):
    NUM_STEP = 300

    og.sim.stop()
    env.reload({"scene": {"type": "Scene"}})
    env.reset()

    results = {field: {} for field in PROFILING_FIELDS}
    for robot_name, robot_cls in tqdm.tqdm(REGISTERED_ROBOTS.items()):
        cur_results = []
        robot = robot_cls(
            prim_path=f"/World/{robot_name}",
            name=robot_name,
            obs_modalities=[]
        )
        og.sim.import_object(robot)
        og.sim.play()
        og.sim.step()

        for _ in range(NUM_STEP):
            cur_result = env.step(np.random.uniform(-0.1, 0.1, robot.action_dim))
            cur_results.append(cur_result[4])
        cur_results = np.array(cur_results)
        for i, field in enumerate(PROFILING_FIELDS):
            results[field][robot_name] = [
                np.mean(cur_results[-100:, i]), np.std(cur_results[-100:, i]), np.median(cur_results[-100:, i]), np.max(cur_results[-100:, i]), np.min(cur_results[-100:, i])
            ]
        og.sim.stop()
        og.sim.remove_object(robot)
    
    return results



def benchmark_scenes(env):
    NUM_STEP = 600

    og.sim.stop()

    results = {field: {} for field in PROFILING_FIELDS}
    for scene_model in tqdm.tqdm(get_available_og_scenes()[12:13]):
        cur_results = []
        env.reload({"scene": {"type": "InteractiveTraversableScene", "scene_model": scene_model}})
        env.reset()

        for _ in range(NUM_STEP):
            cur_result = env.step(None)
            cur_results.append(cur_result[4])
        cur_results = np.array(cur_results)
        for i, field in enumerate(PROFILING_FIELDS):
            results[field][scene_model] = [
                np.mean(cur_results[-100:, i]), np.std(cur_results[-100:, i]), np.median(cur_results[-100:, i]), np.max(cur_results[-100:, i]), np.min(cur_results[-100:, i])
            ]
        og.sim.stop()
    
    return results


def benchmark_particle_system(env): 
    NUM_STEP = 300

    og.sim.stop()
    
    cfg = {"scene": {"type": "Scene"}}

    def check_toggledon(obj):
        return obj.states[object_states.ToggledOn].get_value()

    # Define objects to load into the environment
    sink_cfg = dict(
        type="DatasetObject",
        name="sink",
        category="sink",
        model="yfaufu",
        scale=[0.8, 0.8, 0.8],
        abilities={
            "toggleable": {},
            "particleSource": {
                "conditions": {
                    "water": [check_toggledon],   # Must be toggled on for water source to be active
                },
                "source_radius": 0.0125,
                "source_height": 0.05,
                "initial_speed": 0.0,               # Water merely falls out of the spout
            },
            "particleSink": {
                "conditions": {
                    "water": None,  # No conditions, always sinking nearby particles
                },
                "sink_radius": 0.05,
                "sink_height": 0.05,
            },
        },
        position=[-0.7, 0, 0.56],
    )
    cfg["objects"] = [sink_cfg]
    env.reload(cfg)
    env.reset()
    initial_state = og.sim.dump_state()
    results = {field: {} for field in PROFILING_FIELDS}
    for cond in tqdm.tqdm([True, False]):
        cur_results = []
        gm.ENABLE_OBJECT_STATES = cond
        og.sim.play()
        og.sim.step()

        for _ in range(NUM_STEP):
            cur_result = env.step(None)
            cur_results.append(cur_result[4])
        cur_results = np.array(cur_results)
        for i, field in enumerate(PROFILING_FIELDS):
            results[field][cond] = [
                np.mean(cur_results[-100:, i]), np.std(cur_results[-100:, i]), np.median(cur_results[-100:, i]), np.max(cur_results[-100:, i]), np.min(cur_results[-100:, i])
            ]
        og.sim.stop()
        og.sim.load_state(initial_state)
    
    return results

