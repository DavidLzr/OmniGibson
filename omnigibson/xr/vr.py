import carb
from omni.kit.xr.core import XRCore, XRDeviceClass, XRRay, XRRayQueryResult
from omni.kit.xr.common import XRAvatarManager
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.simulator import Simulator
from typing import Optional, List
import numpy as np
import omnigibson.utils.transform_utils as T
from omnigibson.xr.HTCTrackingSys import HTCTrackingSys

class VRSys():
    def __init__(self, sim: Simulator, use_eye_tracking: bool=False, use_hand_tracking: bool=False) -> None:
        self.sim = sim
        self.xr_core = XRCore.get_singleton()
        self.vr_profile = self.xr_core.get_profile("vr")
        # set empty avatar
        self.vr_profile.set_avatar(XRAvatarManager.get_singleton().create_avatar("empty_avatar", {}))
        # set anchor mode to be scene origin
        carb.settings.get_settings().set(self.vr_profile.get_scene_persistent_path() + "anchorMode", "scene origin")
        # devices info
        self.hmd = None
        self.controllers = {}
        self.trackers = {}
        self.curr_frame_offset = np.zeros(3)
        self.highlighted_focused_object = None
        # HTC tracking system
        self.use_eye_tracking = use_eye_tracking
        self.use_hand_tracking = use_hand_tracking
        self.htcsys = HTCTrackingSys(use_eye_tracking, use_hand_tracking)
    
    def xr2og(self, transform: np.ndarray) -> np.ndarray:
        return transform.T @ T.pose2mat((np.zeros(3), T.euler2quat(np.array([0, np.pi / 2, np.pi / 2]))))
    

    def is_enabled(self) -> bool:
        return self.vr_profile.is_enabled()
    

    def start(self, vr_robot: Optional[BaseRobot]=None) -> None:
        if vr_robot:
            vr_robot.set_position_orientation([0, 0, 0], [0, 0, 0, 1])
            vr_robot.set_params_for_vr()
        self.vr_profile.request_enable_profile()
        self.sim.step()
        assert self.vr_profile.is_enabled(), "VR profile not enabled!"
        # update devices
        self.update_devices()
        # start HTC tracking system
        self.htcsys.start()

    
    def step(self) -> dict:
        vr_data = {}
        # update devices
        self.update_devices()
        # update anchor
        self.update_anchor()
        # get transforms
        vr_data["transforms"] = self.get_transforms()
        # get controller button data
        vr_data["button_data"] = self.get_controller_button_data()
        # step HTC tracking system
        if self.use_eye_tracking:
            vr_data["eye_tracking_data"] = self.get_eye_tracking_data()
            self.highlight_eye_tracking_focused_object(vr_data["eye_tracking_data"])
        if self.use_hand_tracking:
            vr_data["hand_tracking_data"] = self.get_hand_tracking_data()
        return vr_data


    def stop(self) -> None:
        # stop HTC tracking system
        self.htcsys.stop()
        # disable VR profile
        self.xr_core.request_disable_profile()
        self.sim.step()
        assert not self.vr_profile.is_enabled(), "[VRSys] VR profile not disabled!"


    def update_devices(self) -> None:
        for device in self.vr_profile.get_device_list():
            if device.get_class() == XRDeviceClass.xrdisplaydevice:
                self.hmd = device
            elif device.get_class() == XRDeviceClass.xrcontroller:
                self.controllers[device.get_index()] = device
            elif device.get_class() == XRDeviceClass.xrtracker:
                self.trackers[device.get_index()] = device
        assert self.hmd is not None, "[VRSys] HMD not detected! Please make sure you have a VR headset connected to your computer."

    def update_anchor(self) -> None:
        """
        Note: call only once per frame
        """
        # get controller axis input
        offset = np.zeros(3)
        if 1 in self.controllers:
            right_axis_state = self.controllers[1].get_axis_state()
            # calculate right and forward vectors based on hmd transform
            hmd_transform = self.hmd.get_transform()
            right, forward = hmd_transform[0][:3], hmd_transform[2][:3]
            right = right / np.linalg.norm(right)
            forward = forward / np.linalg.norm(forward)
            # calculate offset based on controller input
            offset = np.array([right[i] * right_axis_state["touchpad_x"] - forward[i] * right_axis_state["touchpad_y"] for i in range(3)])
            offset[2] = 0
        if 0 in self.controllers:
            offset[2] = self.controllers[0].get_axis_state()["touchpad_y"]
        
        length = np.linalg.norm(offset)
        if length != 0:
            offset /= length
        # set new anchor transform
        anchorTransform = np.array(self.vr_profile.get_world_anchor_transform())
        self.curr_frame_offset = offset * 0.03
        anchorTransform[3][:3] += self.curr_frame_offset
        self.vr_profile.set_virtual_world_anchor_transform(anchorTransform)


    def get_transforms(self) -> dict:
        """
        Note: because the anchor transform is updated only after sim.step(), we need to manually add the offset to the transform for the current frame
            Therefore this function needs to be called after self.update_anchor()
        """
        transforms = {}
        transforms["hmd"] = self.xr2og(np.array(self.hmd.get_transform()))
        transforms["hmd"][:3, 3] += self.curr_frame_offset
        transforms["controllers"] = {}
        transforms["trackers"] = {}
        for controller_index in self.controllers:
            transforms["controllers"][controller_index] = self.xr2og(np.array(self.controllers[controller_index].get_transform()))
            transforms["controllers"][controller_index][:3, 3] += self.curr_frame_offset
        for tracker_index in self.trackers:
            transforms["trackers"][tracker_index] = self.xr2og(np.array(self.trackers[tracker_index].get_transform()))
            transforms["trackers"][tracker_index][:3, 3] += self.curr_frame_offset
        return transforms


    def get_controller_button_data(self) -> dict:
        button_data = {}
        for controller_index in self.controllers:
            button_data[controller_index] = {}
            button_data[controller_index]["press"] = self.controllers[controller_index].get_button_press_state()
            button_data[controller_index]["touch"] = self.controllers[controller_index].get_button_touch_state()
            button_data[controller_index]["axis"] = self.controllers[controller_index].get_axis_state()
        return button_data
    

    def get_hand_tracking_data(self) -> list:
        return self.htcsys.getHandTrackingData()
    

    def get_eye_tracking_data(self) -> list:
        eye_tracking_data = self.htcsys.getEyeTrackingData()
        # post processing to convert eye tracking data from hmd frame to world frame
        hmd_transform = self.get_transforms()["hmd"]
        if eye_tracking_data[0]:
            eye_tracking_data[3] = T.mat2pose(hmd_transform @ T.pose2mat((eye_tracking_data[3], np.array([0, 0, 0, 1]))))[0]
            eye_tracking_data[4] = hmd_transform[:3, :3] @ eye_tracking_data[4]
            eye_tracking_data[4] /= np.linalg.norm(eye_tracking_data[4])
        if eye_tracking_data[1]:
            eye_tracking_data[5] = T.mat2pose(hmd_transform @ T.pose2mat((eye_tracking_data[5], np.array([0, 0, 0, 1]))))[0]
            eye_tracking_data[6] = hmd_transform[:3, :3] @ eye_tracking_data[6]
            eye_tracking_data[6] /= np.linalg.norm(eye_tracking_data[6])
        if eye_tracking_data[2]:
            eye_tracking_data[7] = T.mat2pose(hmd_transform @ T.pose2mat((eye_tracking_data[7], np.array([0, 0, 0, 1]))))[0]
            eye_tracking_data[8] = hmd_transform[:3, :3] @ eye_tracking_data[8]
            eye_tracking_data[8] /= np.linalg.norm(eye_tracking_data[8])
        return eye_tracking_data
    

    def highlight_eye_tracking_focused_object(self, eye_tracking_data: list) -> None:
        if eye_tracking_data[0]:
            ray = XRRay(origin=eye_tracking_data[3], direction=eye_tracking_data[4])
            self.vr_profile.submit_raycast_query(ray, self.raycast_query_callback)

    def raycast_query_callback(self, ray: XRRay, result: XRRayQueryResult) -> None:
        if result.valid:
            found = False
            usd_target_path = result.get_target_enclosing_model_usd_path()
            print(usd_target_path)
            for object in self.sim.scene.objects:
                if object.prim_path in usd_target_path:
                    found = True
                    if self.highlighted_focused_object is not None and self.highlighted_focused_object != object:
                        self.highlighted_focused_object.highlighted = False
                    if object.category not in ["walls", "floors", "ceilings", "agent"]:
                        object.highlighted = True
                        self.highlighted_focused_object = object
                    break
            if not found and self.highlighted_focused_object:
                self.highlighted_focused_object.highlighted = False
                self.highlighted_focused_object = None
        elif self.highlighted_focused_object:
            self.highlighted_focused_object.highlighted = False
            self.highlighted_focused_object = None
    

