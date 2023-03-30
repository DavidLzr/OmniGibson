#!/usr/bin/env python

import os
import omni
import logging
import argparse
import omnigibson as og
import json

from omnigibson.macros import gm
from omnigibson.examples.profiling.utils import ProfilingEnv, benchmark_objects, benchmark_robots, benchmark_scenes, plot_results

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--all", action='store_true')
parser.add_argument("-s", "--scenes", action='store_true')
parser.add_argument("-o", "--objects", action='store_true')
parser.add_argument("-r", "--robots", action='store_true')


def main():
    args = parser.parse_args()
    # Modify flatcache, pathtracing, GPU, and object state settings
    gm.ENABLE_FLATCACHE = False
    gm.ENABLE_OMNI_PARTICLES = False
    gm.ENABLE_HQ_RENDERING = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.SHOW_DISCLAIMERS = False
    # Disable OmniGibson logging
    log = omni.log.get_log()
    log.enabled = False
    og.log.setLevel(logging.FATAL)

    env = ProfilingEnv(configs=dict(scene={"type": "Scene"}), action_timestep=1/60., physics_timestep=1/240.)
    env.reset()

    results = dict()
    if args.objects or args.all:
        results["objects"] = benchmark_objects(env)
    if args.robots or args.all:
        results["robots"] = benchmark_robots(env)
    if args.scenes or args.all:  
        results["scenes"] = benchmark_scenes(env)

    with open(os.path.join(OUTPUT_DIR, "benchmark_results.json"), 'w') as f:
        json.dump(results, f)
    plot_results(results, os.path.join(OUTPUT_DIR, "omnigibson_benchmark.pdf"))
    og.shutdown()

if __name__ == "__main__":
    main()
