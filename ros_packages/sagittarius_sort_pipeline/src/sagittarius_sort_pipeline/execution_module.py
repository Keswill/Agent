

from typing import Any, Dict, List, Optional, Tuple

import rospy
from sagittarius_object_color_detector.msg import (
    SGRCtrlGoal,
    SGRCtrlResult,
)

from .action_types import Action


Calib = Tuple[float, float, float, float]


_RESULT_NAMES = {
    SGRCtrlResult.SUCCESS: "SUCCESS",
    SGRCtrlResult.ERROR: "ERROR",
    SGRCtrlResult.PREEMPT: "PREEMPT",
    SGRCtrlResult.PLAN_NOT_FOUND: "PLAN_NOT_FOUND",
    SGRCtrlResult.GRASP_FAILD: "GRASP_FAILD",
}


def _result_name(res) -> str:
    if res is None:
        return "NO_RESULT (action server gave nothing back, e.g. timeout)"
    return _RESULT_NAMES.get(res.result, "UNKNOWN({})".format(res.result))


class ExecutionModule:




    BINS: Dict[str, Dict[str, float]] = {
        "LEFT_BIN":  {"x": 0.18, "y":  0.18},
        "RIGHT_BIN": {"x": 0.26, "y":  0.18},
        "FRONT_BIN": {"x": 0.18, "y": -0.15},
    }

    PICK_Z: float = 0.015
    PUT_Z: float = 0.20
    PITCH: float = 1.57
    GOAL_TIMEOUT_SEC: float = 30.0

    def __init__(self, client, calib: Calib):
        self.client = client
        self.k1, self.b1, self.k2, self.b2 = calib


        self._held: Optional[str] = None

        self._last_pose: Optional[Tuple[float, float, float]] = None





    def _pixel_to_arm(self, u: float, v: float) -> Tuple[float, float]:
        return self.k1 * v + self.b1, self.k2 * u + self.b2

    def _send(self, goal: SGRCtrlGoal):
        self.client.send_goal_and_wait(
            goal, rospy.Duration.from_sec(self.GOAL_TIMEOUT_SEC))
        return self.client.get_result()

    def _build_pick_goal(self, u: float, v: float) -> Tuple[SGRCtrlGoal, Tuple[float, float, float]]:
        goal = SGRCtrlGoal()
        goal.action_type = SGRCtrlGoal.ACTION_TYPE_PICK_XYZ
        goal.grasp_type = SGRCtrlGoal.GRASP_OPEN
        goal.pos_x, goal.pos_y = self._pixel_to_arm(u, v)
        goal.pos_z = self.PICK_Z
        goal.pos_pitch = self.PITCH
        return goal, (goal.pos_x, goal.pos_y, goal.pos_z)

    def _build_place_goal(self, bin_name: str) -> Tuple[Optional[SGRCtrlGoal],
                                                        Optional[Tuple[float, float, float]]]:
        if bin_name not in self.BINS:
            return None, None
        b = self.BINS[bin_name]
        goal = SGRCtrlGoal()
        goal.action_type = SGRCtrlGoal.ACTION_TYPE_PUT_XYZ
        goal.pos_x = b["x"]
        goal.pos_y = b["y"]
        goal.pos_z = self.PUT_Z
        goal.pos_pitch = self.PITCH
        return goal, (goal.pos_x, goal.pos_y, goal.pos_z)





    def execute(self, plan: List[Action],
                objects: Dict[str, Dict[str, Any]]) -> None:
        plan = list(plan)
        total = len(plan)
        rospy.loginfo("[exec] ===== starting plan with %d action(s) =====", total)
        rospy.loginfo("[exec] bins: %s", self.BINS)
        rospy.loginfo("[exec] PICK_Z=%.3f  PUT_Z=%.3f  PITCH=%.3f",
                      self.PICK_Z, self.PUT_Z, self.PITCH)

        for i, action in enumerate(plan, start=1):
            tag = "[{}/{}]".format(i, total)
            rospy.loginfo("[exec] -------- %s %s target=%s dest=%s --------",
                          tag, action.kind.upper(), action.target_id,
                          action.destination)

            if action.kind == "pick":
                self._step_pick(tag, action, objects)
            elif action.kind == "place":
                self._step_place(tag, action)
            else:
                rospy.logwarn("[exec] %s unknown action kind %r; skipping",
                              tag, action.kind)

        rospy.loginfo("[exec] ===== plan finished =====")
        if self._held is not None:
            rospy.logwarn(
                "[exec] NOTE at end of plan the tracker still thinks %r is in "
                "the gripper (last place either failed or was skipped)",
                self._held)





    def _step_pick(self, tag: str, action: Action,
                   objects: Dict[str, Dict[str, Any]]) -> None:


        if self._held is not None and self._held != action.target_id:
            rospy.logwarn(
                "[exec] %s WARNING gripper currently holds %r. "
                "sgr_ctrl will open the gripper at the start of this pick goal, "
                "so %r will be dropped at the current arm pose (%s), NOT at any bin.",
                tag, self._held, self._held, self._last_pose)

        if action.target_id not in objects:
            rospy.logwarn("[exec] %s pick target %r not in objects; skipping",
                          tag, action.target_id)
            return
        pos = objects[action.target_id].get("position")
        if pos is None or len(pos) < 2:
            rospy.logwarn("[exec] %s target %r has no pixel position; skipping",
                          tag, action.target_id)
            return

        u, v = pos[0], pos[1]
        goal, (x, y, z) = self._build_pick_goal(u, v)
        rospy.loginfo("[exec] %s PICK %s  pixel=(%.1f, %.1f) -> arm=(%.4f, %.4f, %.4f)",
                      tag, action.target_id, u, v, x, y, z)
        rospy.loginfo("[exec] %s sending PICK_XYZ goal to sgr_ctrl ...", tag)
        res = self._send(goal)
        name = _result_name(res)

        if res is not None and res.result == SGRCtrlResult.SUCCESS:
            rospy.loginfo("[exec] %s PICK %s DONE  result=%s. Gripper now holds %r.",
                          tag, action.target_id, name, action.target_id)
            self._held = action.target_id
            self._last_pose = (x, y, z)
        else:
            rospy.logwarn(
                "[exec] %s PICK %s FAILED  result=%s.",
                tag, action.target_id, name)
            if res is not None and res.result == SGRCtrlResult.PLAN_NOT_FOUND:
                rospy.logwarn(
                    "[exec] %s reason: MoveIt could not find a plan to "
                    "(%.4f, %.4f, %.4f) within %.1fs (or one of the +0.04 / "
                    "+0.12 hover/lift waypoints was unreachable).",
                    tag, x, y, z, 5.0)
            elif res is not None and res.result == SGRCtrlResult.GRASP_FAILD:
                rospy.logwarn(
                    "[exec] %s reason: arm reached target but the servo payload "
                    "reading says the gripper closed on empty air.",
                    tag)




    def _step_place(self, tag: str, action: Action) -> None:
        if self._held is None:
            rospy.logwarn(
                "[exec] %s PLACE %s requested but tracker says nothing is "
                "in the gripper (previous pick probably failed). Skipping "
                "to avoid a pointless motion.",
                tag, action.target_id)
            return

        if self._held != action.target_id:
            rospy.logwarn(
                "[exec] %s PLACE targets %r but gripper actually holds %r "
                "(book-keeping mismatch). Placing the held object anyway.",
                tag, action.target_id, self._held)

        if action.destination is None:
            rospy.logwarn("[exec] %s PLACE %s has no destination; skipping",
                          tag, action.target_id)
            return

        goal, pose = self._build_place_goal(action.destination)
        if goal is None:
            rospy.logerr("[exec] %s unknown bin %r (known: %s); skipping",
                         tag, action.destination, list(self.BINS.keys()))
            return
        x, y, z = pose

        rospy.loginfo("[exec] %s PLACE %s -> %s  arm=(%.4f, %.4f, %.4f)",
                      tag, action.target_id, action.destination, x, y, z)
        rospy.loginfo("[exec] %s sending PUT_XYZ goal to sgr_ctrl ...", tag)
        res = self._send(goal)
        name = _result_name(res)

        if res is not None and res.result == SGRCtrlResult.SUCCESS:
            rospy.loginfo("[exec] %s PLACE %s DONE  result=%s. Gripper is empty.",
                          tag, action.target_id, name)
            self._held = None
            self._last_pose = (x, y, z)
        else:
            rospy.logwarn(
                "[exec] %s PLACE %s FAILED  result=%s.",
                tag, action.target_id, name)
            if res is not None and res.result == SGRCtrlResult.PLAN_NOT_FOUND:
                rospy.logerr(
                    "[exec] %s reason: MoveIt could not plan a path to "
                    "bin %s (%.4f, %.4f, %.4f). The arm did NOT move; it is "
                    "still at the post-pick lift pose holding %r.",
                    tag, action.destination, x, y, z, self._held)
                rospy.logerr(
                    "[exec] %s consequence: the NEXT 'pick' action will open "
                    "the gripper at the start of its own motion, so %r will "
                    "be dropped at the current arm pose (%s) instead of %s. "
                    "Fix: lower PUT_Z (currently %.3f) or move the bin "
                    "closer to the arm base.",
                    tag, self._held, self._last_pose, action.destination,
                    self.PUT_Z)
            elif res is not None and res.result == SGRCtrlResult.GRASP_FAILD:
                rospy.logwarn(
                    "[exec] %s reason: gripper check reported empty on open "
                    "(unusual for PUT_XYZ).",
                    tag)

