# ROS2-Finger-CNN
Code explanation: finger_speed_controller.py

This is a ROS 2 node that counts the fingers on your hand from a webcam and turns that number into a speed for the turtle in turtlesim.

How it works

At start-up the node loads the trained CNN (finger_cnn.keras), opens the camera and creates a publisher on /turtle1/cmd_vel. A timer calls control_loop 10 times per second. On each tick it does the following:

1. It grabs a frame, mirrors it and crops the ROI, a 270x270 pixel square where the hand should be.
2. It converts the crop to grayscale and resizes it to 64x64, exactly as during dataset collection, so the network sees data in the same form it was trained on.
3. The network outputs probabilities for six classes. The node takes the most likely class and its confidence.
4. If the confidence is at least 0.6, the class is added to a history of the last 7 predictions. Otherwise the history is cleared. This filters out random mistakes.
5. The "stable" class is the one that appears most often in the history (Counter.most_common), so a single bad frame does not make the speed jump.
6. The class is converted to a finger value through CLASS_TO_FINGERS. Classes 1 and 2 give -2 and -1 (backward), class 4 gives 1, class 5 gives 2, and classes 0 and 3 give 0. This value is multiplied by SPEED_PER_FINGER to get the speed.
7. The speed is published as a Twist message with linear.x = speed and angular.z = 0, so the turtle only moves forward or backward and never turns.
8. An OpenCV window shows the ROI box, the recognised value and the speed. Q or ESC closes the program.

Safety

The speed is set to zero if a frame cannot be read, the ROI is empty, the confidence is too low or the history is empty. destroy_node also publishes zero speed and releases the camera before exit, so the turtle stops however the program ends.

One thing to watch: the class order printed by the training script must match the keys of CLASS_TO_FINGERS, otherwise the directions will be mixed up.
