#!/usr/bin/env python3

import os
import sys
import time

import cv2
import rospy
import yaml
import actionlib
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal

from sagittarius_sort_pipeline.action_types import parse_plan_strings
from sagittarius_sort_pipeline.execution_module import ExecutionModule
from sagittarius_sort_pipeline.perception_module import PerceptionModule
from sagittarius_sort_pipeline.planning_module import PlanningModule
from sagittarius_sort_pipeline.video_recorder import VideoRecorder


DEFAULT_PLACE_BIN = "LEFT_BIN"


SEARCH_POSE = dict(pos_x=0.2, pos_z=0.15, pos_pitch=1.57)
IMAGE_TOPIC = "/usb_cam/image_raw"
IMAGE_TIMEOUT_SEC = 5.0


def load_calib(path):
    with open(path, "r") as f:
        content = yaml.safe_load(f.read())
    lr = content["LinearRegression"]
    return float(lr["k1"]), float(lr["b1"]), float(lr["k2"]), float(lr["b2"])


def make_search_goal():
    goal = SGRCtrlGoal()
    goal.action_type = SGRCtrlGoal.ACTION_TYPE_XYZ_RPY
    goal.grasp_type = SGRCtrlGoal.GRASP_OPEN
    goal.pos_x = SEARCH_POSE["pos_x"]
    goal.pos_z = SEARCH_POSE["pos_z"]
    goal.pos_pitch = SEARCH_POSE["pos_pitch"]
    return goal


def capture_rgb_frame():
    try:
        msg = rospy.wait_for_message(IMAGE_TOPIC, Image, timeout=IMAGE_TIMEOUT_SEC)
    except rospy.ROSException as e:
        rospy.logerr("no image on %s within %.1fs: %s", IMAGE_TOPIC, IMAGE_TIMEOUT_SEC, e)
        return None
    try:
        bgr = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("cv_bridge error: %s", e)
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_recorder_or_none():
    if not rospy.get_param("~record_video", True):
        rospy.loginfo("[main] recording disabled via ~record_video=false")
        return None

    base_dir = rospy.get_param("~recording_dir", os.path.expanduser("~/.ros/sagittarius_sort_pipeline"))
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, run_tag)

    fps = float(rospy.get_param("~video_fps", 30.0))

    rospy.loginfo("[main] recording to %s (fps=%.1f)", run_dir, fps)
    return VideoRecorder(topic=IMAGE_TOPIC, out_dir=run_dir, fps=fps)


def main():
    rospy.init_node("sort_main", anonymous=False)

    arm_name = rospy.get_param("~arm_name", "sgr532")
    vision_config_path = rospy.get_param("~vision_config")
    instruction = rospy.get_param("~instruction", "sort out all the vegetables")

    rospy.loginfo("[main] instruction: %s", instruction)

    client = actionlib.SimpleActionClient(arm_name + "/sgr_ctrl", SGRCtrlAction)
    rospy.loginfo("[main] waiting for %s/sgr_ctrl action server...", arm_name)
    client.wait_for_server()

    try:
        calib = load_calib(vision_config_path)
    except Exception as e:
        rospy.logerr("[main] cannot load vision_config %r: %s", vision_config_path, e)
        sys.exit(1)
    rospy.loginfo("[main] calib k1=%f b1=%f k2=%f b2=%f", *calib)

    recorder = make_recorder_or_none()
    if recorder is not None:
        recorder.start()

    try:
        rospy.loginfo("[main] moving to search pose...")
        client.send_goal_and_wait(make_search_goal(), rospy.Duration.from_sec(30))

        rgb = capture_rgb_frame()
        if rgb is None:
            rospy.logerr("[main] no frame captured; aborting")
            sys.exit(2)
        rospy.loginfo("[main] captured frame %s", rgb.shape)

        perception = PerceptionModule()
        planning = PlanningModule()
        execution = ExecutionModule(client, calib)

        objects = perception.perceive(rgb)
        rospy.loginfo("[main] perceived %d objects: %s", len(objects), list(objects.keys()))

        plan_strings = planning.plan(instruction, objects)
        rospy.loginfo("[main] planner returned %d raw step(s): %s", len(plan_strings), plan_strings)

        try:
            plan = parse_plan_strings(
                plan_strings, default_place_bin=DEFAULT_PLACE_BIN)
        except ValueError as e:
            rospy.logerr("[main] cannot parse plan: %s", e)
            sys.exit(3)

        rospy.loginfo("[main] parsed %d action(s):", len(plan))
        for i, action in enumerate(plan):
            rospy.loginfo("  %d. %s target=%s dest=%s", i + 1, action.kind, action.target_id, action.destination)

        execution.execute(plan, objects)

        rospy.loginfo("[main] returning to safe pose...")
        safe_goal = SGRCtrlGoal()
        safe_goal.action_type = SGRCtrlGoal.ACTION_TYPE_DEFINE_SAVE
        client.send_goal_and_wait(safe_goal, rospy.Duration.from_sec(30))

        rospy.loginfo("[main] done.")
    finally:
        if recorder is not None:
            try:
                mp4_path = recorder.stop_and_mux("sort.mp4")
                rospy.loginfo("[main] video saved: %s", mp4_path)
            except Exception as e:
                rospy.logerr("[main] recorder shutdown failed: %s", e)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
