import rclpy
from arm_control_pkg.arm_manual import ManualControlNode
from arm_control_pkg.arm_commute_node import ArmCummuteNode
from arm_control_pkg.load_params import LoadParams
from arm_control_pkg.arm_angle_control import ArmAngleControl
from rclpy.executors import MultiThreadedExecutor


def main(args=None):
    rclpy.init(args=args)
    load_params = LoadParams("arm_control_pkg")
    arm_agnle_control = ArmAngleControl(arm_params=load_params)
    arm_commute_node = ArmCummuteNode(
        arm_params=load_params, arm_angle_control=arm_agnle_control
    )
    arm_manual_node = ManualControlNode(
        arm_commute_node=arm_commute_node, arm_angle_control_node=arm_agnle_control
    )
    executor = MultiThreadedExecutor()
    executor.add_node(arm_commute_node)
    executor.add_node(arm_manual_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # action_server.get_logger().info("Keyboard interrupt, shutting down...")
        pass
    finally:
        # action_server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
