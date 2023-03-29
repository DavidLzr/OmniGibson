"""
Example script for vr system.
"""

import omnigibson as og
from omnigibson.xr.vr import VRSys
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.asset_utils import get_available_og_scenes

SCENES = ["empty"] + get_available_og_scenes()

EYE_TRACKING_MODE = dict(
    disabled="Disable eye tracking (default)",
    enabled="Enable eye tracking",
)

VR_ROBOT = dict(
    Behaviorbot="Behaviorbot (default)",
)

def main(random_selection=False):
    scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)
    use_eye_tracking = choose_from_options(options=EYE_TRACKING_MODE, name="eye tracking mode", random_selection=random_selection)
    use_eye_tracking = use_eye_tracking == "enabled" 
    robot_name = choose_from_options(options=VR_ROBOT, name="vr robot", random_selection=random_selection)

    # Create the config for generating the environment we want
    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model
        # scene_cfg["load_object_categories"] = ["floors", "walls"]
    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    
    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    vrsys = VRSys(sim=og.sim, use_eye_tracking=use_eye_tracking, use_hand_tracking=False)
    
    # Update the control mode of the robot
    vr_robot = env.robots[0]
    # Reset environment
    env.reset()
    og.sim.enable_viewer_camera_teleoperation()
    # start vrsys 
    vrsys.start(vr_robot)

    for _ in range(100000):
        if og.sim.is_playing():
            vr_data = vrsys.step()
            action = vr_robot.gen_action_from_vr_data(vr_data)
            env.step(action)                


    # Always shut down the environment cleanly at the end
    env.close()

if __name__ == "__main__":
    main()