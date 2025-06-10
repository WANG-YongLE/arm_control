def solveIK(self, target_pos, weight_constraint=100.0):
    fixed_joint_index = 4  # 固定第5個關節
    fixed_joint_value = np.pi / 2

    variable_indices = [i for i in range(len(self.controllable_joints)) if i != fixed_joint_index]

    def ik_loss(q_var):
        q_full = np.zeros(len(self.controllable_joints))
        for i, idx in enumerate(variable_indices):
            q_full[idx] = q_var[i]
        q_full[fixed_joint_index] = fixed_joint_value

        for i, joint_index in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, joint_index, q_full[i])

        tcp_pos = np.array(p.getLinkState(self.robot_id, self.end_eff_index)[4])
        pos_error = np.linalg.norm(tcp_pos - target_pos)
        penalty = (q_full[0] + q_full[1] + q_full[2] - np.pi / 2) ** 2
        return pos_error + weight_constraint * penalty

    q_init_var = [0.0] * len(variable_indices)

    res = minimize(
        ik_loss,
        q_init_var,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * len(q_init_var)
    )

    if res.success:
        q_solution = np.zeros(len(self.controllable_joints))
        for i, idx in enumerate(variable_indices):
            q_solution[idx] = res.x[i]
        q_solution[fixed_joint_index] = fixed_joint_value

        print("✅ IK求解成功")
        print("Joint角度（degrees）:", np.rad2deg(q_solution))
        self.setJointPosition(q_solution)
        return q_solution
    else:
        print("❌ IK求解失敗")
        return None