

import os
import threading
import time

try:
    import queue as _queue
except ImportError:
    import Queue as _queue

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class VideoRecorder:
    def __init__(self,
                 topic: str,
                 out_dir: str,
                 fps: float = 30.0,
                 queue_maxsize: int = 60,
                 jpeg_quality: int = 90):
        self.topic = topic
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)

        os.makedirs(self.frames_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._queue = _queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._worker = None
        self._sub = None


        self._received = 0
        self._written = 0
        self._dropped = 0
        self._frame_size = None
        self._lock = threading.Lock()





    def start(self) -> None:
        if self._sub is not None:
            rospy.logwarn("[recorder] already started; ignoring second start()")
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name="sort_video_recorder", daemon=True)
        self._worker.start()
        self._sub = rospy.Subscriber(
            self.topic, Image, self._callback, queue_size=5)
        rospy.loginfo("[recorder] started on %s -> %s (fps tag=%.1f)",
                      self.topic, self.frames_dir, self.fps)

    def stop_and_mux(self, mp4_name: str = "sort.mp4") -> str:
        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                rospy.logwarn(
                    "[recorder] worker did not exit within 10s; "
                    "continuing to mux whatever was written.")
            self._worker = None

        with self._lock:
            received, written, dropped = (
                self._received, self._written, self._dropped)
            size = self._frame_size
        rospy.loginfo(
            "[recorder] stopped. received=%d written=%d dropped=%d",
            received, written, dropped)

        mp4_path = os.path.join(self.out_dir, mp4_name)
        if written == 0 or size is None:
            rospy.logwarn("[recorder] no frames captured; skipping mp4 mux.")
            return mp4_path

        self._mux_mp4(mp4_path, size)
        return mp4_path





    def _callback(self, msg: Image) -> None:
        if self._stop_event.is_set():
            return
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "[recorder] cv_bridge error: %s", e)
            return
        with self._lock:
            self._received += 1
            if self._frame_size is None:
                h, w = bgr.shape[:2]
                self._frame_size = (int(w), int(h))
            idx = self._received
        try:
            self._queue.put_nowait((idx, bgr))
        except _queue.Full:
            with self._lock:
                self._dropped += 1
            rospy.logwarn_throttle(
                2.0, "[recorder] queue full; dropped frame (%d so far)",
                self._dropped)





    def _worker_loop(self) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while True:
            try:
                idx, bgr = self._queue.get(timeout=0.2)
            except _queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            path = os.path.join(self.frames_dir, "{:06d}.jpg".format(idx))
            try:
                ok = cv2.imwrite(path, bgr, params)
                if ok:
                    with self._lock:
                        self._written += 1
                else:
                    rospy.logwarn_throttle(
                        5.0, "[recorder] cv2.imwrite returned False for %s", path)
            except Exception as e:
                rospy.logwarn_throttle(
                    5.0, "[recorder] imwrite error %s: %s", path, e)





    def _mux_mp4(self, mp4_path, frame_size) -> None:
        frames = sorted(
            f for f in os.listdir(self.frames_dir)
            if f.lower().endswith(".jpg"))
        if not frames:
            rospy.logwarn("[recorder] no jpg frames found under %s",
                          self.frames_dir)
            return



        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, self.fps, frame_size)
        if not writer.isOpened():
            rospy.logerr("[recorder] cv2.VideoWriter failed to open %s", mp4_path)
            return

        t0 = time.time()
        for i, name in enumerate(frames, start=1):
            img = cv2.imread(os.path.join(self.frames_dir, name))
            if img is None:
                rospy.logwarn_throttle(5.0, "[recorder] cannot read %s", name)
                continue
            if (img.shape[1], img.shape[0]) != frame_size:
                img = cv2.resize(img, frame_size)
            writer.write(img)
        writer.release()
        dt = time.time() - t0
        rospy.loginfo(
            "[recorder] wrote %s (%d frames, %.1f fps tag, %.1fs to mux)",
            mp4_path, len(frames), self.fps, dt)
