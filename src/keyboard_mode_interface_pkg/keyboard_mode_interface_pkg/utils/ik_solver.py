# ############################################
# A Robot controller for kinematics, dynamics
# and control based on pyBullet framework
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.1
# ############################################

# Input:
# 1. robot_type: specify urdf file initials eg. if urdf file name is 'ur5.urdf', specify 'ur5'
# 2. controllable_joints: joint indices of controllable joints. If not specified, by default all joints indices except first joint (first joint is fixed joint between robot stand and base)
# 3. end-eff_index: specify the joint indices for end-effector link. If not specified, by default the last controllable_joints is considered as end-effector joint
# 4. time_Step: time step for simulation

import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import os
from ament_index_python.packages import get_package_share_directory
import xml.etree.ElementTree as ET
import math
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
class PybulletRobotController:
    def __init__(
        self,
        
        initial_height=0,

        controllable_joints=None,
        end_eff_index=None,
        time_step=1e-3,
    ):
        initial_height=0
        robot_description_path = get_package_share_directory("robot_description")
        self.urdf_path = os.path.join(robot_description_path, "urdf", "target.urdf")
        self.robot_id = None
        self.num_joints = None
        self.controllable_joints = controllable_joints
        self.end_eff_index = end_eff_index
        self.time_step = time_step
        self.previous_ee_position = None
        self.initial_height = initial_height  # 新增的高度參數
        
        # 讀取並初始化關節限制
        self.joint_limits = self.get_joint_limits_from_urdf()
        self.num_joints = len(self.joint_limits)  # 使用關節數量設定
        self.box_id = None  # 初始化 box_id 為 None
    # function to initiate pybullet and engine and create world
    def createWorld(self, GUI=True, view_world=False):
        # load pybullet physics engine
        if GUI:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        GRAVITY = -9.8
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10
        )
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf")
        rotation = R.from_euler("z", 0, degrees=True).as_quat()
        planeId = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, planeId)

        # loading robot into the environment
        
        self.robot_id = p.loadURDF(
            self.urdf_path,
            useFixedBase=True,
            basePosition=[0, 0, self.initial_height],
            baseOrientation=rotation,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_MERGE_FIXED_LINKS,
        )

        self.num_joints = p.getNumJoints(self.robot_id)  # Joints
        print("#Joints:", self.num_joints)
        if self.controllable_joints is None:
            self.controllable_joints = list(range( self.num_joints-3))
        print("#Controllable Joints:", self.controllable_joints)
        print("#End-effector:", self.end_eff_index)
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"總關節數量: {self.num_joints}")
        self.controllable_joints = list(range( self.num_joints))
        print(f"可控制的關節索引: {self.controllable_joints}")
        print(f"需要提供的初始位置數量: {len(self.controllable_joints)}")
        self.markEndEffector()
        cubeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        cubePos = [5, 5, 5]
        cubeOri = p.getQuaternionFromEuler([0, 0, 0])
        p.createMultiBody(1, cubeId, -1, cubePos, cubeOri)

        if view_world:
            while True:
                p.stepSimulation()
                time.sleep(self.time_step)

    # function to joint position, velocity and torque feedback

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    # function for setting joint positions of robot
    def setJointPosition(self, position, kp=1.0, kv=1.0):
        # print('Joint position controller')

        zero_vec = [0.0] * len(self.controllable_joints)
        # print(f"控制的關節索引數量: {len(self.controllable_joints)}")
        print(f"輸入的目標位置數量: {(position)}")

        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=position,
            targetVelocities=zero_vec,
            positionGains=[kp] * len(self.controllable_joints),
            velocityGains=[kv] * len(self.controllable_joints),
        )
        for _ in range(20):  # to settle the robot to its position
            p.stepSimulation()

    def get_base_position(self):
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        return base_position

    def get_joint_limits_from_urdf(self):
        """
        從 URDF 文件中讀取每個關節的範圍限制。

        Returns:
            joint_limits (dict): 包含每個關節的最小和最大角度限制。
        """
        joint_limits = {}
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            joint_name = joint.get("name")

            # 忽略特定的夹具关节，比如 "gripper_joint"

            joint_type = joint.get("type")
            if joint_type == "revolute" or joint_type == "continuous":
                limit = joint.find("limit")
                if limit is not None:
                    lower = float(
                        limit.get("lower", -6.28318)
                    )  # 預設為 -180 度（以 radians 為單位）
                    upper = float(limit.get("upper", 6.28318))  # 預設為 180 度
                    joint_limits[joint_name] = (lower, upper)
                elif joint_type == "prismatic":
                    limit = joint.find("limit")
                    if limit is not None:
                        lower = float(limit.get("lower", -0.01))
                        upper = float(limit.get("upper", 0.01))
                        joint_limits[joint_name] = (lower, upper)
        return joint_limits

    def get_current_pose(self, link_index=None):
        """
        根據指定的手臂關節角度來獲取特定連結的世界座標和旋轉矩陣。

        Args:
            link_index (int, optional): 連結索引。默認為倒數第一個連結（即末端執行器）。

        Returns:
            position (np.array): 指定連結的世界坐標 [x, y, z]。
            rotation_matrix (np.array): 指定連結的旋轉矩陣 (3x3)。
        """
        # 獲取機器人的總連結數
        num_links = p.getNumJoints(self.robot_id)

        # 處理負索引，使其轉換為對應的正索引
        if link_index is None:
            link_index = self.end_eff_index  # 默認為末端執行器
        elif link_index < 0:
            link_index = num_links + link_index  # 將負索引轉換為正索引

        # 驗證link_index是否合法
        if link_index < 0 or link_index >= num_links:
            raise ValueError(
                f"Invalid link_index {link_index}. Valid range is 0 to {num_links - 1}."
            )

        # 獲取指定連結的鏈接狀態
        link_state = p.getLinkState(
            self.robot_id, link_index, computeForwardKinematics=True
        )

        # 取得指定連結在世界座標系中的位置和方向（四元數）
        link_position = np.array(link_state[4])  # worldLinkFramePosition
        link_orientation = link_state[5]  # worldLinkFrameOrientation (四元數)

        # 將四元數轉換為旋轉矩陣
        rotation_matrix = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape(
            3, 3
        )
        return link_position, rotation_matrix


    def get_base_pose(self):
        # 抓基座世界座標
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)

        base_position = np.array(base_position)

        # 返回基座位置和方向
        return base_position, base_orientation

  
    def markTarget(self, target_position):
        # 使用紅色標記顯示目標位置
        line_length = 0.1  # 調整標記大小
        p.addUserDebugLine(
            [target_position[0] - line_length, target_position[1], target_position[2]],
            [target_position[0] + line_length, target_position[1], target_position[2]],
            [1, 0, 0],  # 紅色
            lineWidth=3,
        )
        p.addUserDebugLine(
            [target_position[0], target_position[1] - line_length, target_position[2]],
            [target_position[0], target_position[1] + line_length, target_position[2]],
            [1, 0, 0],
            lineWidth=3,
        )
        p.addUserDebugLine(
            [target_position[0], target_position[1], target_position[2] - line_length],
            [target_position[0], target_position[1], target_position[2] + line_length],
            [1, 0, 0],
            lineWidth=3,
        )

    # function to solve forward kinematics
    def solveForwardPositonKinematics(self):
        # get end-effector link state
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
        # eePose = list(link_trn) + list(p.getEulerFromQuaternion(link_rot))
        eePose = list(frame_pos) + list(p.getEulerFromQuaternion(frame_rot))
        print("now End-effector pose:", eePose)
        return eePose

    def format_joint_angles(joint_angles, precision=3):
        """
        將列表中的所有角度轉換為 float，並保留小數點後指定位數。

        Args:
            joint_angles (list): 原始的關節角度列表。
            precision (int): 保留的小數位數 (預設為 3)。

        Returns:
            list: 格式化後的關節角度列表。
        """
        return [round(float(angle), precision) for angle in joint_angles]



    # function to solve inverse kinematics
    # 單獨使用要少取一個 因為會輸出 6
    def solveInversePositionKinematics(self, end_eff_pose):
        """
        計算逆向運動學以獲取關節角度，基於給定的末端執行器姿勢。

        Args:
            end_eff_pose (list): 末端執行器的目標位置和姿勢，
                                格式為 [x, y, z, roll, pitch, yaw] (6 個元素) 或 [x, y, z] (3 個元素)。

        Returns:
            list: 對應的關節角度。
        """
        print(f"end_eff_pose:{end_eff_pose}")
        if len(end_eff_pose) == 6:
            joint_angles = p.calculateInverseKinematics(
                self.robot_id,
                self.end_eff_index,
                targetPosition=end_eff_pose[0:3],
                targetOrientation=p.getQuaternionFromEuler(end_eff_pose[3:6]),
            )
        else:
            joint_angles = p.calculateInverseKinematics(
                self.robot_id, self.end_eff_index, targetPosition=end_eff_pose[0:3]
            )

        # 標記末端執行器的位置路徑
        self.markEndEffectorPath()
        return joint_angles

    def markEndEffector(self):
        # 獲取末端執行器的位置
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        ee_position = eeState[0]  # 末端執行器的位置
        print("End-effector position:", ee_position)
        # 使用藍色點標記末端執行器位置
        p.addUserDebugPoints(
            pointPositions=[list(ee_position)], 
            pointColorsRGB=[[0, 0, 1]],  # 把顏色改成列表的列表
            pointSize=5
        )


    def markEndEffectorPath(self):
        # 獲取當前末端執行器的位置
        eeState = p.getLinkState(self.robot_id, self.end_eff_index)
        ee_position = eeState[0]

        # 如果是第一個點，設置 previous_ee_position
        if self.previous_ee_position is None:
            self.previous_ee_position = ee_position

        # 繪製從上次位置到當前位置的線
        p.addUserDebugLine(
            self.previous_ee_position,
            ee_position,
            lineColorRGB=[0, 0, 1],  # 藍色
            lineWidth=2,
        )

        # 更新 previous_ee_position 為當前位置
        self.previous_ee_position = ee_position



    def generate_random_target_and_solve_ik(self, x_range=(-0.15, -0.45), y_range=(-0.15,-0.45), z_range=(0.2, 0.6 ), steps=30):
        """
        Generates a random target point, marks it, calculates the IK solution,
        and generates a smooth angle sequence to reach it.

        Args:
            x_range (tuple): Min/max range for x coordinate.
            y_range (tuple): Min/max range for y coordinate.
            z_range (tuple): Min/max range for z coordinate.
            steps (int): Number of steps for smooth transition.

        Returns:
            list or None: A sequence of joint angles for the transition if successful, otherwise None.
        """
        # Generate random target position
        target_x = random.uniform(x_range[0], x_range[1])
        target_y = random.uniform(y_range[0], y_range[1])
        target_z = random.uniform(z_range[0], z_range[1])+self.initial_height

        target_position = [target_x, target_y, target_z]
        roll = math.pi/2
        pitch = 0
        yaw = 0
        target_position += [roll, pitch, yaw]
        print(f"Generated random target: {target_position}")

        # Mark the target position
        self.markTarget(target_position)

        # Calculate IK solution for the target position
        target_joint_angles = self.solveInversePositionKinematics(target_position)

        if target_joint_angles and len(target_joint_angles) >= len(self.controllable_joints):
            target_joint_angles = np.array(target_joint_angles[:len(self.controllable_joints)])
            print(f"IK solution found: {target_joint_angles.tolist()}")

            # Get current joint positions
            current_positions = np.array(self.getJointStates()[0])

            angle_sequence = []
            # Generate smooth transition sequence
            for step in range(steps):
                t = (step + 1) / steps # Start from t > 0 to move towards target
                intermediate_positions = (1 - t) * current_positions + t * target_joint_angles
                angle_sequence.append(intermediate_positions.tolist())

            return angle_sequence
        else:
            print("Could not find a valid IK solution for the random target.")
            return None
    def go_to_position(self,position, steps=10):

        target_joint_angles = self.solveInversePositionKinematics(position)

        if target_joint_angles and len(target_joint_angles) >= len(self.controllable_joints):
            target_joint_angles = np.array(target_joint_angles[:len(self.controllable_joints)])
           # print(f"IK solution found: {target_joint_angles.tolist()}")

            # Get current joint positions
            current_positions = np.array(self.getJointStates()[0])

            angle_sequence = []
            # Generate smooth transition sequence
            for step in range(steps):
                t = (step + 1) / steps # Start from t > 0 to move towards target
                intermediate_positions = (1 - t) * current_positions + t * target_joint_angles
                angle_sequence.append(intermediate_positions.tolist())

            return angle_sequence
        else:
            print("Could not find a valid IK solution for the random target.")
            return None

    def remove_flat_box(self):
        """
        刪除之前建立的方塊物件（若存在）
        """
        if self.box_id is not None:
            p.removeBody(self.box_id)
            self.box_id = None
            print("已成功刪除方塊。")
        else:
            print("尚未建立方塊，無需刪除。")

    def create_flat_box_from_points(self,p0, p1, color=[1, 0, 0, 1]):
        """
        使用兩個 3D 點建立一個對齊線段的長方體，垂直向下延伸到底部平面。
        
        :param p0: numpy array, 第 0 個點 (3D)
        :param p1: numpy array, 第 1 個點 (3D)
        :param color: 可選，RGBA 顏色
        """
        # 計算 ab 向量（只考慮 XY 平面方向）
        ab_vec = p1[:2] - p0[:2]
        ab_len = np.linalg.norm(ab_vec)
        ab_dir = ab_vec / ab_len
        self.markTarget(p0)
        # 垂直方向
        normal_dir = np.array([-ab_dir[1], ab_dir[0]])

        # 方塊尺寸
        box_length = 0.08
        box_width = 0.50
        box_height = p0[2]  # 向下延伸到地面

        # 中心位置（XY）
        center_xy = p0[:2] + ab_dir * (box_length / 2) + normal_dir * (box_width / 2)
        center_z = box_height / 2
        center_pos = [center_xy[0], center_xy[1], center_z]

        # 姿態（yaw 角轉為四元數）
        yaw = np.arctan2(ab_dir[1], ab_dir[0])
        orientation = p.getQuaternionFromEuler([0, 0, yaw])

        # 半尺寸
        half_extents = [box_length / 2, box_width / 2, box_height / 2]

        # 建立形狀
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)

        if self.box_id is None:
            # 第一次建立
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            self.box_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=center_pos,
                baseOrientation=orientation
            )
            return self.box_id
        else:
            # 已建立，直接移動
            p.resetBasePositionAndOrientation(self.box_id, center_pos, orientation)
            return self.box_id
    def has_collision(self):
        return p.getContactPoints(bodyA=self.robot_id)

