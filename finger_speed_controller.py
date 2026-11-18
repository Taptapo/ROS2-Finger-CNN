from collections import Counter, deque

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist
from tensorflow import keras


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "finger_cnn.keras"

ROI = (350, 50, 620, 320)

IMG_SIZE = 64

CONF_THRESHOLD = 0.6

SMOOTH_WINDOW = 7

SPEED_PER_FINGER = 1.0

PUBLISH_RATE = 10.0

# Class index -> number of fingers
# Must match the order printed by your training script.

CLASS_TO_FINGERS = {
    0: 0,
    1: -2,
    2: -1,
    3: 0,
    4: 1,
    5: 2
}


# ==========================================
# ROS 2 NODE
# ==========================================

class FingerSpeedController(Node):

    def __init__(self):
        super().__init__("finger_speed_controller")

        # Load CNN
        self.get_logger().info("Loading CNN model...")

        self.model = keras.models.load_model(MODEL_PATH)

        self.get_logger().info("Model loaded!")

        # Open camera
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # ROS 2 publisher
        self.publisher = self.create_publisher(
            Twist,
            "/turtle1/cmd_vel",
            10
        )

        # Prediction history
        self.history = deque(maxlen=SMOOTH_WINDOW)

        # Current speed
        self.current_speed = 0.0

        # Timer
        self.timer = self.create_timer(
            1.0 / PUBLISH_RATE,
            self.control_loop
        )

        self.get_logger().info("Finger speed controller started!")

    # ======================================
    # PUBLISH SPEED
    # ======================================

    def publish_speed(self, speed):

        msg = Twist()

        msg.linear.x = float(speed)
        msg.angular.z = 0.0

        self.publisher.publish(msg)

    # ======================================
    # CAMERA + CNN + ROS 2
    # ======================================

    def control_loop(self):

        ok, frame = self.cap.read()

        if not ok:
            self.current_speed = 0.0
            self.history.clear()
            self.publish_speed(0.0)
            return

        # Mirror camera image
        frame = cv2.flip(frame, 1)

        # Extract ROI
        x1, y1, x2, y2 = ROI

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            self.current_speed = 0.0
            self.history.clear()
            self.publish_speed(0.0)
            return

        # Preprocessing (same as your prediction script)
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        sample = cv2.resize(
            gray,
            (IMG_SIZE, IMG_SIZE)
        )

        batch = sample.reshape(
            1, IMG_SIZE, IMG_SIZE, 1
        ).astype("float32")

        # CNN prediction
        probs = self.model.predict(
            batch,
            verbose=0
        )[0]

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # Confidence check
        if conf >= CONF_THRESHOLD:
            self.history.append(pred)
        else:
            self.history.clear()

        # Stable prediction
        stable = (
            Counter(self.history).most_common(1)[0][0]
            if self.history else None
        )

        # Calculate speed
        if stable is not None and stable in CLASS_TO_FINGERS:

            fingers = CLASS_TO_FINGERS[stable]

            self.current_speed = fingers * SPEED_PER_FINGER

        else:
            self.current_speed = 0.0

        # Publish ROS 2 velocity
        self.publish_speed(self.current_speed)

        # ==================================
        # DISPLAY WINDOW
        # ==================================

        # Green ROI rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Finger prediction
        text = (
            f"Fingers: {stable} ({conf * 100:.0f}%)"
            if stable is not None
            else "Recognizing..."
        )

        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        # Speed display
        cv2.putText(
            frame,
            f"Speed: {self.current_speed:.1f} units/s",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        # Show camera window
        cv2.imshow(
            "CNN finger recognition",
            frame
        )

        # Exit on Q or ESC
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            self.get_logger().info("Closing controller...")
            rclpy.shutdown()

    # ======================================
    # CLEANUP
    # ======================================

    def destroy_node(self):

        self.publish_speed(0.0)

        self.cap.release()

        cv2.destroyAllWindows()

        super().destroy_node()


# ==========================================
# MAIN
# ==========================================

def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = FingerSpeedController()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
