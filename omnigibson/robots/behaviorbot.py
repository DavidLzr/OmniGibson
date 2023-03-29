import os
import itertools
from typing import List, Tuple
from collections import OrderedDict
from abc import ABC

import numpy as np
from scipy.spatial.transform import Rotation as R
from pxr import Gf
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.manipulation_robot import ManipulationRobot, GraspingPoint
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.objects.usd_object import USDObject
import omnigibson.utils.transform_utils as T
from omnigibson.utils.constants import JointType

COMPONENT_SUFFIXES = ['x', 'y', 'z', 'rx', 'ry', 'rz']

HEAD_TO_BODY_OFFSET = ([0, 0, -0.4], [0, 0, 0, 1])
RH_TO_BODY_OFFSET = ([0, 0.15, -0.4], T.euler2quat([0, 160, -80]))
LH_TO_BODY_OFFSET = ([0, -0.15, -0.4], T.euler2quat([0, 160, 80]))

# Body parameters
BODY_HEIGHT_RANGE = (0, 2)  # meters. The body is allowed to clip the floor by about a half.
BODY_MASS = 30  # body mass in kg
BODY_MOVING_FORCE = BODY_MASS * 500

# Hand parameters
HAND_GHOST_HAND_APPEAR_THRESHOLD = 0.15
THUMB_2_POS = [0, -0.02, -0.05]
THUMB_1_POS = [0, -0.015, -0.02]
PALM_CENTER_POS = [0, -0.04, 0.01]
PALM_BASE_POS = [0, 0, 0.015]
FINGER_TIP_POS = [0, -0.025, -0.055]
HAND_LIFTING_FORCE = 300

# Assisted grasping parameters
ASSIST_FRACTION = 1.0
ARTICULATED_ASSIST_FRACTION = 0.7
MIN_ASSIST_FORCE = 0
MAX_ASSIST_FORCE = 500
ASSIST_FORCE = MIN_ASSIST_FORCE + (MAX_ASSIST_FORCE - MIN_ASSIST_FORCE) * ASSIST_FRACTION
TRIGGER_FRACTION_THRESHOLD = 0.5
CONSTRAINT_VIOLATION_THRESHOLD = 0.1
ATTACHMENT_BUTTON_TIME_THRESHOLD = 1

# Hand link index constants
PALM_LINK_NAME = "palm"
FINGER_MID_LINK_NAMES = ("Tproximal", "Iproximal", "Mproximal", "Rproximal", "Pproximal")
FINGER_TIP_LINK_NAMES = ("Tmiddle", "Imiddle", "Mmiddle", "Rmiddle", "Pmiddle")
THUMB_LINK_NAME = "Tmiddle"
HAND_SKELETON_INDEX = [[2, 5, 9, 13, 17], [3, 6, 10, 14, 18]]

# Gripper parameters
GRIPPER_GHOST_HAND_APPEAR_THRESHOLD = 0.25
GRIPPER_JOINT_POSITIONS = [0.550569, 0.000000, 0.549657, 0.000000]

# Head parameters
HEAD_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HEAD_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HEAD_DISTANCE_THRESHOLD = 0.5  # distance threshold in meters


class Behaviorbot(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    A humanoid robot that can be used in VR as an avatar. It has two hands, a body and a head with two cameras.
    """

    def __init__(
            self,
            # Shared kwargs in hierarchy
            name,
            prim_path=None,
            class_id=None,
            uuid=None,
            scale=None,
            visible=True,
            fixed_base=True,
            visual_only=False,
            self_collisions=True,
            load_config=None,

            # Unique to USDObject hierarchy
            abilities=None,

            # Unique to ControllableObject hierarchy
            control_freq=None,
            controller_config=None,
            action_type="continuous",
            action_normalize=False,
            reset_joint_pos=None,

            # Unique to BaseRobot
            obs_modalities="rgb",
            proprio_obs="default",

            # Unique to ManipulationRobot
            grasping_mode="assisted",

            # unique to Behaviorbot
            agent_id=1,
            use_body=True,
            use_ghost_hands=True,

            **kwargs
    ):
        """
        Initializes Behaviorbot
        Args:
            agent_id (int): unique id of the agent - used in multi-user VR
            image_width (int): width of each camera
            image_height (int): height of each camera
            use_body (bool): whether to use body
            use_ghost_hands (bool): whether to use ghost hand
        """

        super(Behaviorbot, self).__init__(
            prim_path=prim_path,
            name=name,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            grasping_mode=grasping_mode,
            **kwargs,
        )

        # Basic parameters
        self.agent_id = agent_id
        self.use_body = use_body
        self.use_ghost_hands = use_ghost_hands
        self._world_base_fixed_joint_prim = None

        # Activation parameters
        self.first_frame = True
        # whether hand or body is in contact with other objects (we need this since checking contact list is costly)
        self.part_is_in_contact = {hand_name: False for hand_name in self.arm_names + ["body"]}

        # Whether the VR system is actively hooked up to the VR agent.
        self.vr_attached = False
        self._vr_attachment_button_press_timestamp = None

        # setup eef parts
        self.parts = OrderedDict()
        self.parts["lh"] = BRPart(
            name="lh", parent=self, prim_path="lh_palm", eef_type="hand",
            offset_to_body=LH_TO_BODY_OFFSET, **kwargs
        )

        self.parts["rh"] = BRPart(
            name="rh", parent=self, prim_path="rh_palm", eef_type="hand",
            offset_to_body=RH_TO_BODY_OFFSET, **kwargs
        )

        self.parts["head"] = BRPart(
            name="head", parent=self,  prim_path="head", eef_type="head",
            offset_to_body=HEAD_TO_BODY_OFFSET, **kwargs
        )

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/behaviorbot/usd/Behaviorbot.usd")

    @property
    def model_name(self):
        return "Behaviorbot"
    
    @property
    def n_arms(self):
        return 2
    
    @property
    def arm_names(self):
        return ["lh", "rh"]
    
    @property
    def eef_link_names(self):
        dic = {arm: f"{arm}_{PALM_LINK_NAME}" for arm in self.arm_names}
        dic["head"] = "head"
        return dic

    @property
    def arm_link_names(self):
        """The head counts as a arm since it has the same 33 joint configuration"""
        return {arm: [f"{arm}_{component}" for component in COMPONENT_SUFFIXES] for arm in self.arm_names + ['head']}
    
    @property
    def finger_link_names(self):
        return {
            arm: [f"{arm}_{link_name}" for link_name in itertools.chain(FINGER_MID_LINK_NAMES, FINGER_TIP_LINK_NAMES)]
            for arm in self.arm_names
        }
    
    @property
    def base_joint_names(self):
        return [f"base_{component}_joint" for component in COMPONENT_SUFFIXES]
    
    @property
    def arm_joint_names(self):
        """The head counts as a arm since it has the same 33 joint configuration"""
        return {eef: [f"{eef}_{component}_joint" for component in COMPONENT_SUFFIXES] for eef in self.arm_names + ["head"]}
    
    @property
    def finger_joint_names(self):
        return {
            arm: (
                # palm-to-proximal joints.
                [f"{arm}_{to_link}__{arm}_{PALM_LINK_NAME}" for to_link in FINGER_MID_LINK_NAMES]
                +
                # proximal-to-tip joints.
                [f"{arm}_{to_link}__{arm}_{from_link}" for from_link, to_link in zip(FINGER_MID_LINK_NAMES, FINGER_TIP_LINK_NAMES)]
            )
            for arm in self.arm_names
        }
    
    @property
    def base_control_idx(self):
        # TODO: might need to refactor out joints
        joints = list(self.joints.keys())
        return tuple(joints.index(joint) for joint in self.base_joint_names)
    
    @property
    def arm_control_idx(self):
        joints = list(self.joints.keys())
        return {
            arm: [joints.index(f"{arm}_{component}_joint") for component in COMPONENT_SUFFIXES]
            for arm in self.arm_names
        }

    @property
    def gripper_control_idx(self):
        joints = list(self.joints.values())
        return {arm: [joints.index(joint) for joint in arm_joints] for arm, arm_joints in self.finger_joints.items()}

    @property
    def camera_control_idx(self):
        joints = list(self.joints.keys())
        return [joints.index(f"head_{component}_joint") for component in COMPONENT_SUFFIXES]

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def controller_order(self):
        controllers = ["base", "camera"]
        for arm_name in self.arm_names:
            controllers += [f"arm_{arm_name}", f"gripper_{arm_name}"]
        return controllers

    @property
    def _default_controllers(self):
        controllers = {
            "base": "JointController",
            "camera": "JointController"
        }
        controllers.update({f"arm_{arm_name}": "JointController" for arm_name in self.arm_names})
        controllers.update({f"gripper_{arm_name}": "MultiFingerGripperController" for arm_name in self.arm_names})
        return controllers

    @property
    def _default_base_joint_controller_config(self):
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": False,
            "motor_type": "position",
            "dof_idx": self.base_control_idx,
            "command_input_limits": None,
            "command_output_limits": None,
        }

    @property
    def _default_arm_joint_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_input_limits": None,
                "command_output_limits": None,  # TODO: Set this to "default" after control limits are fixed.
                "use_delta_commands": False,
            }
        return dic
    
    @property
    def _default_gripper_multi_finger_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "MultiFingerGripperController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "inverted": True,
                "mode": "smooth",
            }
        return dic
    
    @property
    def _default_camera_joint_controller_config(self):
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
            "command_input_limits": None,
            "command_output_limits": None,
            "use_delta_commands": False,
        }

    @property
    def _default_controller_config(self):
        controllers = {
            "base": {"JointController": self._default_base_joint_controller_config},
            "camera": {"JointController": self._default_camera_joint_controller_config},
        }
        controllers.update(
            {
                f"arm_{arm_name}": {"JointController": self._default_arm_joint_controller_configs[arm_name]}
                for arm_name in self.arm_names
            }
        )
        controllers.update(
            {
                f"gripper_{arm_name}": {
                    "MultiFingerGripperController": self._default_gripper_multi_finger_controller_configs[arm_name],
                    "JointController": self._default_gripper_joint_controller_configs[arm_name],
                }
                for arm_name in self.arm_names
            }
        )
        return controllers

    def load(self):
        prim = super(Behaviorbot, self).load()
        for part in self.parts.values():
            part.load()
        return prim

    def _post_load(self):
        super()._post_load()

    def _create_discrete_action_space(self):
        raise ValueError("Behaviorbot does not support discrete actions!")

    def set_params_for_vr(self):
        # set joint mass and rigid body properties
        for link in self.links.values():
            link.ccd_enabled = True
        for arm in self.arm_names:
            for link in self.finger_link_names[arm]:
                self.links[link].mass = 0.005
        self.links["base"].mass = 15
        self.links["lh_palm"].mass = 0.12
        self.links["rh_palm"].mass = 0.12

        
        
        # set base joint properties
        for joint_name in self.base_joint_names:
            self.joints[joint_name].stiffness = 1e8
            self.joints[joint_name].max_effort = 7500

        # set arm joint properties
        for arm in self.arm_joint_names:
            for joint_name in self.arm_joint_names[arm]:
                self.joints[joint_name].stiffness = 1e8
                self.joints[joint_name].max_effort = 300

        # set finger joint properties
        for arm in self.finger_joint_names:
            for joint_name in self.finger_joint_names[arm]:
                self.joints[joint_name].max_effort = 1e3
                self.joints[joint_name].stiffness = 1e8

    @property
    def base_footprint_link_name(self):
        """
        Name of the actual root link that we are interested in. 
        """
        return "base"

    @property
    def base_footprint_link(self):
        """
        Returns:
            RigidPrim: base footprint link of this object prim
        """
        return self._links[self.base_footprint_link_name]


    def get_position_orientation(self):
        # If the simulator is playing, return the pose of the base_footprint link frame
        if self._dc is not None and self._dc.is_simulating():
            return self.base_footprint_link.get_position_orientation()

        # Else, return the pose of the robot frame
        else:
            return super().get_position_orientation()

    def set_position_orientation(self, position=None, orientation=None):
        super().set_position_orientation(position, orientation)
        # Move the joint frame for the world_base_joint
        if self._world_base_fixed_joint_prim is not None:
            if position is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
            if orientation is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*np.float_(orientation)[[3, 0, 1, 2]]))

    @property
    def assisted_grasp_start_points(self):
        side_coefficients = {"lh": np.array([1, -1, 1]), "rh": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name=f"{arm}_{PALM_LINK_NAME}", position=PALM_BASE_POS),
                GraspingPoint(link_name=f"{arm}_{PALM_LINK_NAME}", position=PALM_CENTER_POS * side_coefficients[arm]),
                GraspingPoint(
                    link_name=f"{arm}_{THUMB_LINK_NAME}", position=THUMB_1_POS * side_coefficients[arm]
                ),
                GraspingPoint(
                    link_name=f"{arm}_{THUMB_LINK_NAME}", position=THUMB_2_POS * side_coefficients[arm]
                ),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        side_coefficients = {"lh": np.array([1, -1, 1]), "rh": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name=f"{arm}_{finger}", position=FINGER_TIP_POS * side_coefficients[arm])
                for finger in FINGER_TIP_LINK_NAMES
            ]
            for arm in self.arm_names
        }

    def update_hand_contact_info(self):
        self.part_is_in_contact["body"] = len(self.links["body"].contact_list())
        for hand_name in self.arm_names:
            self.part_is_in_contact[hand_name] = len(self.eef_links[hand_name].contact_list()) \
               or np.any([len(finger.contact_list()) for finger in self.finger_links[hand_name]])
    
    def gen_action_from_vr_data(self, vr_data):
        """
        Generates an action for the Behaviorbot to perform based on vr data dict.

        Action space (all non-normalized values that will be clipped if they are too large)
        * See Behaviorbot.py for details on the clipping thresholds for
        Body:
        - 6DOF pose delta - relative to body frame from previous frame
        Eye:
        - 6DOF pose delta - relative to body frame (where the body will be after applying this frame's action)
        Left hand, right hand (in that order):
        - 6DOF pose delta - relative to body frame (same as above)
        - Trigger fraction

        Total size: 26
        """
        # Actions are stored as 1D numpy array
        action = np.zeros(26)
        action[[18, 25]] = 1 # trigger frac defaults to 1 (hands open)

        # Update body action space
        hmd_transform = vr_data["transforms"]["hmd"].copy()
        prev_body_pos, prev_body_orn = self.get_position_orientation()
        prev_body_rpy = R.from_quat(prev_body_orn).as_euler("XYZ")
        if np.all(np.equal(hmd_transform, np.identity(4))):
            delta_body_pos, delta_body_rpy = np.zeros(3), np.zeros(3)
        else:
            des_body_pos, des_body_orn = T.mat2pose(hmd_transform)
            des_body_pos[2] -= 0.45
            des_body_rpy = np.array([0, 0, R.from_quat(des_body_orn).as_euler("XYZ")[2]])
            des_body_orn = T.euler2quat(des_body_rpy)
            delta_body_pos = des_body_pos - prev_body_pos
            delta_body_rpy = np.arctan2(np.sin(des_body_rpy - prev_body_rpy), np.cos(des_body_rpy - prev_body_rpy))
        action[self.controller_action_idx["base"]] = np.concatenate([des_body_pos, des_body_rpy])

        # # Update action space for other VR objects
        for part_name, eef_part in self.parts.items():
            # Process local transform adjustments
            prev_local_pos, prev_local_orn = eef_part.local_position_orientation
            prev_local_rpy = R.from_quat(prev_local_orn).as_euler("XYZ")
            if part_name == "head":
                part_transform = vr_data["transforms"]["hmd"]
                trigger_frac = None
            elif part_name == "lh":
                part_transform = vr_data["transforms"]["controllers"][0]
                trigger_frac = vr_data["button_data"][0]["axis"]["trigger"]
                reset = vr_data["button_data"][0]["press"]["grip"]
            elif part_name == "rh":
                part_transform = vr_data["transforms"]["controllers"][1]
                trigger_frac = vr_data["button_data"][1]["axis"]["trigger"]
                reset = vr_data["button_data"][1]["press"]["grip"]

            if np.all(np.equal(part_transform, np.identity(4))):
                des_world_part_pos, des_world_part_orn = prev_local_pos, prev_local_orn
            else:
                # apply eef rotation offset to part transform first
                part_transform = part_transform @ T.pose2mat((np.zeros(3), eef_part.offset_to_body[1]))
                des_world_part_pos, des_world_part_orn = T.mat2pose(part_transform)

            # Get local pose with respect to the new body frame
            des_local_part_pos, des_local_part_orn = T.relative_pose_transform(
                des_world_part_pos, des_world_part_orn, des_body_pos, des_body_orn
            )
            # apply shoulder position offset to the part transform to get final destination pose
            des_local_part_pos, des_local_part_orn = T.pose_transform(
                eef_part.offset_to_body[0], [0, 0, 0, 1], des_local_part_pos, des_local_part_orn
            )

            des_part_rpy = R.from_quat(des_local_part_orn).as_euler("XYZ")
            delta_pos = des_local_part_pos - prev_local_pos
            delta_rpy = np.arctan2(np.sin(des_part_rpy - prev_local_rpy), np.cos(des_part_rpy - prev_local_rpy))
            controller_name = "camera" if part_name == "head" else "arm_" + part_name
            action[self.controller_action_idx[controller_name]] = np.concatenate([des_local_part_pos, des_part_rpy])
            # Process trigger fraction and reset for controllers
            if trigger_frac:
                # Map from (0, 1) to (-1, 1) range, inverted.
                action[self.controller_action_idx[f"gripper_{part_name}"]] = -2 * trigger_frac + 1 

            # If we reset, action is 1, otherwise 0
            if reset:
                self.parts[part_name].set_position_orientation(des_local_part_pos, des_part_rpy)
            # update ghost hand if necessary
            if self.use_ghost_hands and part_name is not "head":
                self.parts[part_name].update_ghost_hands(des_world_part_pos, des_world_part_orn)
        return action


class BRPart(ABC):
    """This is the interface that all Behaviorbot eef parts must implement."""

    def __init__(self, name: str, parent: Behaviorbot, prim_path: str, eef_type: str, offset_to_body: List[float]) -> None:
        """
        Create an object instance with the minimum information of class ID and rendering parameters.

        Args:
            name (str): unique name of this BR part
            parent (Behaviorbot): the parent BR object
            prim_path (str): prim path to the root link of the eef
            eef_type (str): type of eef. One of hand, head
            offset_to_body (List[float]): relative rotation offset between the shoulder_rz link and the eef link.
        """
        self.name = name
        self.parent = parent
        self.prim_path = prim_path
        self.eef_type = eef_type
        self.offset_to_body = offset_to_body

        self.ghost_hand = None
        self._root_link = None
        self._world_position_orientation = ([0, 0, 0], [0, 0, 0, 1])

    def load(self) -> None:
        self._root_link = self.parent.links[self.prim_path]
        # setup ghost hand
        if self.eef_type == "hand":
            gh_name = f"ghost_hand_{self.name}"
            self.ghost_hand = USDObject(
                prim_path=f"/World/{gh_name}",
                usd_path=os.path.join(assets_path, f"models/behaviorbot/usd/{gh_name}.usd"),
                name=gh_name,
                scale=0.001,
                visible=False,
                visual_only=True,
            )
            og.sim.import_object(self.ghost_hand)

    @property
    def local_position_orientation(self) -> Tuple[List[float], List[float]]:
        """
        Get local position and orientation w.r.t. to the body
        Return:
            Tuple[Array[x, y, z], Array[x, y, z, w]]

        """
        return T.relative_pose_transform(*self.world_position_orientation, *self.parent.get_position_orientation())

    @property
    def world_position_orientation(self) -> None:
        """
        Get position and orientation in the world space
        Return:
            Tuple[Array[x, y, z], Array[x, y, z, w]]
        """
        return self._root_link.get_position_orientation()

    def set_position_orientation(self, pos: List[float], orn: List[float]) -> None:
        """
        Call back function to set the base's position
        """
        self.parent.joints[f"{self.name}_x_joint"].set_pos(pos[0], drive=False)
        self.parent.joints[f"{self.name}_y_joint"].set_pos(pos[1], drive=False)
        self.parent.joints[f"{self.name}_z_joint"].set_pos(pos[2], drive=False)
        self.parent.joints[f"{self.name}_rx_joint"].set_pos(orn[0], drive=False)
        self.parent.joints[f"{self.name}_ry_joint"].set_pos(orn[1], drive=False)
        self.parent.joints[f"{self.name}_rz_joint"].set_pos(orn[2], drive=False)

    def update_ghost_hands(self, pos: List[float], orn: List[float]) -> None:
        """
        Updates ghost hand to track real hand and displays it if the real and virtual hands are too far apart.
        Args:
            pos (List[float]): list of positions [x, y, z]
            orn (List[float]): list of rotations [x, y, z, w]
        """
        assert self.eef_type == "hand", "ghost hand is only valid for BR hand!"
        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position_orientation(pos, orn)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = np.linalg.norm(pos - self.world_position_orientation[0])
        should_visible = dist_to_real_controller > HAND_GHOST_HAND_APPEAR_THRESHOLD

        # Only toggle visibility if we are transition from hidden to unhidden, or the other way around
        if self.ghost_hand.visible is not should_visible:
            self.ghost_hand.visible = should_visible
