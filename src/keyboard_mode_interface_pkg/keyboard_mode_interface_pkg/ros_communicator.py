from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PointStamped
from std_msgs.msg import String, Header,Float32MultiArray
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu
from trajectory_msgs.msg import JointTrajectoryPoint
from .model import DeviceDataTypeEnum
from nav2_msgs.srv import ClearEntireCostmap
from visualization_msgs.msg import Marker

class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")
        self.publisher_joint_trajectory = self.create_publisher(
            JointTrajectoryPoint, DeviceDataTypeEnum.robot_arm, 10
        )
        self.latest_amcl_pose = None  # 存儲最新的AMCL位姿數據
        self.subscriber_amcl = self.create_subscription(#改成實際機械手臂位姿
            PoseWithCovarianceStamped,  # 消息類型：帶協方差的位姿
            "/amcl_pose",  # 話題名稱
            self.subscriber_amcl_callback,  # 回調函數
            10  # 隊列大小
        )
        self.target_pose = None
        self.subscriber_goal = self.create_subscription(
            PoseStamped, "/goal_pose", self.subscriber_goal_callback, 1
        )
        self.latest_received_global_plan = None
        self.subscriber_received_global_plan = self.create_subscription(
            Path, "/received_global_plan", self.received_global_plan_callback, 1
        )
        self.latest_yolo_coordinates = None
        self.subscriber_yolo_detection_position = self.create_subscription(
            PointStamped,
            "/yolo/detection/position",
            self.yolo_detection_position_callback,
            10,
        )
        # Subscribe to YOLO detected object coordinates
        self.latest_yolo_offset = None
        self.subscriber_yolo_offset = self.create_subscription(
            PointStamped,
            "/yolo/detection/offset",
            self.yolo_detection_offset_callback,
            10,
        )
        self.latest_imu_data = None
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self.imu_data_callback, 10
        )

        self.latest_mediapipe_data = None
        self.mediapipe_sub = self.create_subscription(
            Point, "/mediapipe_data", self.mediapipe_data_callback, 10
        )

        self.latest_yolo_target_info = None
        self.yolo_target_info_sub = self.create_subscription(
            Float32MultiArray, "/yolo/target_info", self.yolo_target_info_callback, 10
        )

        self.latest_camera_x_multi_depth = None
        self.camera_x_multi_depth_sub = self.create_subscription(
            Float32MultiArray,
            "/camera/x_multi_depth_values",
            self.camera_x_multi_depth_callback,
            10,
        )

        # publish goal_pose
        self.publisher_goal_pose = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # publish robot arm angle
        self.publisher_joint_trajectory = self.create_publisher(
            JointTrajectoryPoint, DeviceDataTypeEnum.robot_arm, 10
        )

        self.publisher_coordinates = self.create_publisher(
            PointStamped, "/coordinates", 10
        )

        self.publisher_target_label = self.create_publisher(String, "/target_label", 10)

        self.crane_state_publisher = self.create_publisher(String, "crane_state", 10)

        self.publisher_confirmed_path = self.create_publisher(
            Path, "/confirmed_initial_plan", 10
        )

        self.publisher_target_marker = self.create_publisher(
            Marker, "/selected_target_marker", 10
        )
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear"
        )
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear"
        )

        self.publisher_received_global_plan = self.create_publisher(
            Path, "/received_global_plan", 10
        )
        self.publisher_plan = self.create_publisher(Path, "/plan", 10)
    # publish robot arm angle


    def clear_received_global_plan(self):
        """
        清空 /received_global_plan 话题
        """
        empty_path = Path()
        empty_path.header.frame_id = "map"
        self.publisher_received_global_plan.publish(empty_path)
        self.get_logger().info("Published empty Path to /received_global_plan")

    def clear_plan(self):
        """
        清空 /plan 话题
        """
        empty_path = Path()
        empty_path.header.frame_id = "map"
        self.publisher_plan.publish(empty_path)
        self.get_logger().info("Published empty Path to /plan")

    def subscriber_amcl_callback(self, msg):
        self.latest_amcl_pose = msg

    def get_latest_amcl_pose(self):
        if self.latest_amcl_pose is None:
            self.get_logger().warn("No AMCL pose data received yet.")
        return self.latest_amcl_pose

    # goal callback and get_latest_goal
    def subscriber_goal_callback(self, msg):
        position = msg.pose.position
        target = [position.x, position.y, position.z]
        self.target_pose = target

    def get_latest_goal(self):
        if self.target_pose is None:
            self.get_logger().warn("No goal pose data received yet.")
        return self.target_pose    
    def received_global_plan_callback(self, msg):
        self.latest_received_global_plan = msg

    def get_latest_received_global_plan(self):
        if self.latest_received_global_plan is None:
            self.get_logger().warn("No received global plan data received yet.")
            return None
        return self.latest_received_global_plan
    def publish_goal_pose(self, goal):
        goal_pose = PoseStamped()
        goal_pose.header = Header()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = goal[0]
        goal_pose.pose.position.y = goal[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        self.publisher_goal_pose.publish(goal_pose)
    def publish_coordinates(self, x, y, z, frame_id="map"):
        coordinate_msg = PointStamped()
        coordinate_msg.header.stamp = self.get_clock().now().to_msg()
        coordinate_msg.header.frame_id = frame_id
        coordinate_msg.point.x = x
        coordinate_msg.point.y = y
        coordinate_msg.point.z = z
        self.publisher_coordinates.publish(coordinate_msg)

    def mediapipe_data_callback(self, msg):
        self.latest_mediapipe_data = msg

    def get_latest_mediapipe_data(self):
        if self.latest_mediapipe_data is None:
            self.get_logger().warn("No Mediapipe data received yet.")
            return None
        return self.latest_mediapipe_data

    def yolo_target_info_callback(self, msg):
        self.latest_yolo_target_info = msg

    def get_latest_yolo_target_info(self):
        if self.latest_yolo_target_info is None:
            return None
        return self.latest_yolo_target_info

    def camera_x_multi_depth_callback(self, msg):
        self.latest_camera_x_multi_depth = msg

    def get_latest_camera_x_multi_depth(self):
        if self.latest_camera_x_multi_depth is None:
            return None
        return self.latest_camera_x_multi_depth
    def imu_data_callback(self, msg):
        self.latest_imu_data = msg

    def get_latest_imu_data(self):
        if self.latest_imu_data is None:
            return None
        return self.latest_imu_data
    # YOLO coordinates callback
    def yolo_detection_position_callback(self, msg):
        """Callback to receive YOLO detected object coordinates."""
        self.latest_yolo_coordinates = msg

    def get_latest_yolo_detection_position(self):
        """Getter for the latest YOLO detected object coordinates."""
        if self.latest_yolo_coordinates is None:
            return None
        return self.latest_yolo_coordinates

    def yolo_detection_offset_callback(self, msg):
        self.latest_yolo_offset = msg

    def get_latest_yolo_detection_offset(self):
        if self.latest_yolo_offset is None:
            return None
        return self.latest_yolo_offset

    def publish_target_label(self, label):
        target_label_msg = String()
        target_label_msg.data = label
        self.publisher_target_label.publish(target_label_msg)
    def publish_confirmed_initial_plan(self, path_msg: Path):
        """
        確認路徑使用
        """
        self.publisher_confirmed_path.publish(path_msg)

    def publish_selected_target_marker(self, x, y, z=0.0):
        """
        在 foxglove 畫紅點
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 0.2  # 球體大小
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # 透明度
        marker.color.r = 1.0  # 顏色
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher_target_marker.publish(marker)
    def publish_robot_arm_angle(self, angle):
        joint_trajectory_point = JointTrajectoryPoint()
        joint_trajectory_point.positions = angle
        joint_trajectory_point.velocities = [0.0] * len(angle)
        self.publisher_joint_trajectory.publish(joint_trajectory_point)



