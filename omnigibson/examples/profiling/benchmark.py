#!/usr/bin/env python

import os
import argparse
import json
import omnigibson as og
import numpy as np

from omnigibson.macros import gm
from omnigibson.utils.profiling_utils import ProfilingEnv

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--robot", action='store_true')
parser.add_argument("-s", "--scene", action='store_true')
parser.add_argument("-f", "--flatcache", action='store_true')
parser.add_argument("-p", "--particles", action='store_true')

PROFILING_FIELDS = ["Total step fps", "Action generation fps", "Physics step fps", "Render step time", "Non physics step fps"]

def main():
    args = parser.parse_args()
    # Modify flatcache, pathtracing, GPU, and object state settings
    gm.ENABLE_FLATCACHE = args.flatcache
    gm.ENABLE_HQ_RENDERING = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.ENABLE_GPU_DYNAMICS = True

    cfg = {}
    if args.robot:
        cfg["robots"] =  [{
            "type": "Fetch",
            "obs_modalities": ["scan", "rgb", "depth"],
            "action_type": "continuous",
            "action_normalize": True,
            "controller_config": {"arm_0": {"name": "JointController"}}
        }]
    if args.particles:
        pass
    if args.scene:
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        }
    else:
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        }
    env = ProfilingEnv(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    env.reset()

    output, results = [], []
    for i in range(500):
        if args.robot:
            result = env.step(np.random.uniform(-0.1, 0.1, env.robots[0].action_dim))[4][:5]
        else:
            result = env.step(None)[4][:5]
        results.append(result)

    results = np.array(results)
    for i, field in enumerate(PROFILING_FIELDS):
        field += " (Rs_int, " if args.scene else " (Rs_int, "
        field += "with Fetch, " if args.robot else "without Fetch, "
        field += "with particles & softbody " if args.flatcache else "without particles & softbody "
        field += "flatcache on)" if args.flatcache else "flatcache off)"
        output.append({
            "name": field,
            "unit": "fps",
            "value": 1 / np.mean(results[-200:, i])
        })

    with open("output.json", "w") as f:
        json.dump(output, f)
    og.shutdown()

if __name__ == "__main__":
    main()
