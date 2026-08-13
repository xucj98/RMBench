from ._base_task import Base_Task
from .utils import *

class rearrange_blocks(Base_Task):
    
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.25, -0.15],
            ylim=[-0.2, -0.1],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button.set_mass(0.0001, ['button_cap'])
        self.set_button_unpressed(self.button)
        self.press_cnt = 0
        self.press_flag = False

        def create_block(block_pose):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=(1,0,0),
                name="box",
            )
        
        def create_mat(mat_pose):
            return create_box(
                scene=self,
                pose=mat_pose,
                half_size=mat_half_size,
                color=(0.000, 0.502, 0.996),
                name="box",
                is_static=True,
            )
        
        self.block_half_size = 0.02
        blocks_pose = []
        x_block = 0.02
        self.block_y_lim = np.random.uniform(-0.15, -0.08)
        for _ in range(3):
            block_pos = rand_pose(
                xlim=[x_block, x_block],
                ylim=[self.block_y_lim, self.block_y_lim],
                zlim=[0.741+self.block_half_size],
                qpos=[1, 0, 0, 0],
            )
            blocks_pose.append(block_pos)
            x_block += 0.12
        selected_mat_block = np.random.choice([0, 2])
        self.block1 = create_block(blocks_pose[1])
        self.block2 = create_block(blocks_pose[selected_mat_block])
        
        mat_half_size = [0.04, 0.04, 0.0005]
        mats_pose = []
        x_mat = 0.02
        for _ in range(2):
            mat_pos = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[self.block_y_lim, self.block_y_lim],
                zlim=[0.7415],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            x_mat += 0.24
        self.mat1 = create_mat(mats_pose[0])
        self.mat2 = create_mat(mats_pose[1])
        if selected_mat_block == 0:
            empty_mat = 2
            self.final_occupy_mat = self.mat2
            self.mid_mat = self.mat1
        else:
            empty_mat = 0
            self.final_occupy_mat = self.mat1
            self.mid_mat = self.mat2
        self.target_pose1 = blocks_pose[empty_mat]
        self.target_pose2 = blocks_pose[1]
        self.stage_id = 0
        self._oracle_key_state_phase = 0

        self.first_empty_mat_name = ['left', 'null', 'right'][empty_mat]
        self.second_block_name = ['right', 'null', 'left'][empty_mat]
        self.initial_occupied_mat_side = self.second_block_name
        self._diagnostic_first_placement_ever_ready = False
        self._diagnostic_second_placement_ever_ready = False
        self._diagnostic_min_button_value = 0.0
    
    def play_once(self):
        start = self._key_state_stage_start()
        self.move(self.grasp_actor(self.block1,arm_tag="right",pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the {self.first_empty_mat_name} block and move it to the {self.first_empty_mat_name} empty mat.')
        self._record_key_state_micro_stage("block1_pick", start)
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the {self.first_empty_mat_name} block and move it to the {self.first_empty_mat_name} empty mat.')
        self._record_key_state_micro_stage("block1_lift", start)
        start = self._key_state_stage_start()
        self.move(self.place_actor(self.block1, arm_tag="right", target_pose=self.target_pose1, functional_point_id=2, dis=0.01), language_annotation=f'Pick up the {self.first_empty_mat_name} block and move it to the {self.first_empty_mat_name} empty mat.')
        self._record_key_state_micro_stage("block1_place", start)
        start = self._key_state_stage_start()
        self.press_button()
        self._record_key_state_micro_stage("press_button", start)
        start = self._key_state_stage_start()
        self.move(self.back_to_origin(arm_tag="left"), language_annotation=f'Press button.')
        self._record_key_state_micro_stage("press_return", start)
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self._record_key_state_micro_stage("block2_prepare", start)
        start = self._key_state_stage_start()
        self.move(self.grasp_actor(self.block2,arm_tag="right",pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self._record_key_state_micro_stage("block2_pick", start)
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self._record_key_state_micro_stage("block2_lift", start)
        start = self._key_state_stage_start()
        self.move(self.place_actor(self.block2, arm_tag="right", target_pose=self.target_pose2, functional_point_id=2, dis=0.01), language_annotation=f'Pick up the {self.second_block_name} block and move it between the two mats.')
        self._record_key_state_micro_stage("block2_place", start)
        self._set_key_state_scene_info(
            task_facts={
                "empty_mat_side": self.first_empty_mat_name,
                "initial_occupied_mat_side": self.initial_occupied_mat_side,
            },
            phase_sequence=[
                "move_middle_block_to_empty_mat",
                "press_button_after_first_move",
                "move_original_mat_block_to_middle",
            ],
        )
        return self.info
        
    def press_button(self):
        self.move(
            self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0),
            self.back_to_origin(arm_tag="right"),
            language_annotation=f'Press button.')
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=f'Press button.')
        self.check_press_success()
        self.check_success()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=f'Press button.')
        self.set_button_unpressed(self.button)
        self.update_button_reset(self.button)
    
    def get_current_button_value(self, button_name, joint_name="button_joint", target=0.0):
        if button_name == 'button':
            button_actor = self.button
        else:
            button_actor = self.check_button
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor      
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        return qpos[idx]
    
    def set_button_unpressed(self, button_actor, joint_name="button_joint", target=0.0): 
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor   
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name)
        qpos = art.get_qpos()
        qpos[idx] = target   
        art.set_qpos(qpos) 
        joints[idx].set_drive_target(target)

    def check_button_pressed(self, button_actor, joint_name="button_joint", threshold=-0.005): 
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor   
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name) 
        qpos = art.get_qpos()
        if qpos[idx] < threshold:
            return True
        else:
            return False
    
    def update_button_reset(self, button_actor, joint_name="button_joint", threshold=-0.001): 
        art = button_actor.actor if hasattr(button_actor, "actor") else button_actor   
        joints = art.get_active_joints()   
        joint_names = [j.get_name() for j in joints]    
        idx = joint_names.index(joint_name) 
        qpos = art.get_qpos()
        if qpos[idx] > threshold:
            self.press_flag = False
    
    def check_press_success(self):
        button_value = float(self.get_current_button_value("button"))
        self._diagnostic_min_button_value = min(
            self._diagnostic_min_button_value,
            button_value,
        )
        if button_value < -0.005 and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1
            snapshot = self._get_rearrange_diagnostic_snapshot()
            self._record_eval_diagnostic_event(
                "button_pressed",
                press_count=self.press_cnt,
                stage_id=self.stage_id,
                conditions=snapshot["conditions"],
                metrics=snapshot["metrics"],
            )

    def _get_rearrange_diagnostic_snapshot(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        final_occupy_mat_pose = self.final_occupy_mat.get_pose().p
        mid_occupy_mat_pose = self.mid_mat.get_pose().p
        empty_region_center = [0.13, self.block_y_lim]

        block1_target_dx = float(np.abs(block1_pose[0] - final_occupy_mat_pose[0]))
        block1_target_dy = float(np.abs(block1_pose[1] - final_occupy_mat_pose[1]))
        block2_original_dx = float(np.abs(block2_pose[0] - mid_occupy_mat_pose[0]))
        block2_original_dy = float(np.abs(block2_pose[1] - mid_occupy_mat_pose[1]))
        block2_middle_dx = float(np.abs(block2_pose[0] - empty_region_center[0]))
        block2_middle_dy = float(np.abs(block2_pose[1] - empty_region_center[1]))
        button_value = float(self.get_current_button_value("button"))

        conditions = {
            "block1_on_target_mat": block1_target_dx < 0.03 and block1_target_dy < 0.03,
            "block1_below_height_limit": bool(block1_pose[2] < 0.77),
            "block2_on_original_mat": block2_original_dx < 0.03 and block2_original_dy < 0.03,
            "block2_below_height_limit": bool(block2_pose[2] < 0.77),
            "block2_in_middle": block2_middle_dx < 0.03 and block2_middle_dy < 0.03,
            "right_gripper_open": bool(self.is_right_gripper_open()),
            "button_latched_pressed": bool(self.press_flag),
        }
        conditions["first_placement_ready"] = bool(
            conditions["block1_on_target_mat"]
            and conditions["block1_below_height_limit"]
            and conditions["block2_on_original_mat"]
            and conditions["block2_below_height_limit"]
        )
        conditions["second_placement_ready"] = bool(
            conditions["block1_on_target_mat"]
            and conditions["block2_in_middle"]
            and conditions["right_gripper_open"]
            and not conditions["button_latched_pressed"]
        )
        return {
            "conditions": conditions,
            "metrics": {
                "button_joint_position": button_value,
                "block1_target_dx": block1_target_dx,
                "block1_target_dy": block1_target_dy,
                "block1_z": float(block1_pose[2]),
                "block2_original_dx": block2_original_dx,
                "block2_original_dy": block2_original_dy,
                "block2_middle_dx": block2_middle_dx,
                "block2_middle_dy": block2_middle_dy,
                "block2_z": float(block2_pose[2]),
            },
        }

    def _update_rearrange_diagnostic_progress(self, snapshot):
        conditions = snapshot["conditions"]
        if conditions["first_placement_ready"] and not self._diagnostic_first_placement_ever_ready:
            self._diagnostic_first_placement_ever_ready = True
            self._record_eval_diagnostic_event(
                "first_placement_ready",
                stage_id=self.stage_id,
                conditions=conditions,
                metrics=snapshot["metrics"],
            )
        if (
            self.stage_id >= 1
            and conditions["second_placement_ready"]
            and not self._diagnostic_second_placement_ever_ready
        ):
            self._diagnostic_second_placement_ever_ready = True
            self._record_eval_diagnostic_event(
                "second_placement_ready",
                stage_id=self.stage_id,
                conditions=conditions,
                metrics=snapshot["metrics"],
            )

    def check_success(self):
        if self.stage_id == 2:
            self.max_reward = max(self.max_reward, 1.0)
            return True
        if self.press_cnt > 1:
            return False
        self.update_button_reset(self.button)
        self.check_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button")+0.002))

        snapshot = self._get_rearrange_diagnostic_snapshot()
        self._update_rearrange_diagnostic_progress(snapshot)
        conditions = snapshot["conditions"]

        if self.press_flag: # all on mat
            if self.stage_id == 0 and self.press_cnt == 1 and conditions["first_placement_ready"]:
                self.stage_id = 1
                self._record_eval_diagnostic_event(
                    "stage_transition",
                    from_stage=0,
                    to_stage=1,
                    conditions=conditions,
                    metrics=snapshot["metrics"],
                )
            return False
        else:
            if self.stage_id == 1 and self.press_cnt == 1 and conditions["second_placement_ready"]:
                self.stage_id = 2
                self.max_reward = max(self.max_reward, 1.0)
                self._record_eval_diagnostic_event(
                    "stage_transition",
                    from_stage=1,
                    to_stage=2,
                    conditions=conditions,
                    metrics=snapshot["metrics"],
                )
                return True
            return False

    def _oracle_left_arm_returned(self):
        current = np.asarray(self.get_arm_pose("left"), dtype=np.float32)[:3]
        target = np.asarray(self.robot.left_original_pose, dtype=np.float32)[:3]
        return bool(np.linalg.norm(current - target) < 0.04)

    def get_oracle_key_state(self):
        """Return the current structured memory from simulator task state.

        The phase is monotonic. P1 starts after the first placement is
        physically complete and released; P2 starts after one valid button
        press has been released and the left arm has returned. This mirrors the query-time state semantics
        used by the state-token dataset without using future trajectory data.
        """
        snapshot = self._get_rearrange_diagnostic_snapshot()
        conditions = snapshot["conditions"]
        if (
            self._oracle_key_state_phase == 0
            and conditions["first_placement_ready"]
            and conditions["right_gripper_open"]
        ):
            self._oracle_key_state_phase = 1
        if (
            self._oracle_key_state_phase == 1
            and self.stage_id >= 1
            and self.press_cnt == 1
            and not self.press_flag
            and self._oracle_left_arm_returned()
        ):
            self._oracle_key_state_phase = 2

        phase_labels = [
            "move_middle_block_to_empty_mat",
            "press_button_after_first_move",
            "move_original_mat_block_to_middle",
        ]
        phase = phase_labels[self._oracle_key_state_phase]
        if self._oracle_key_state_phase != 1:
            button_status = "NA"
        elif self.press_cnt == 1:
            button_status = "confirmed"
        else:
            button_status = "unconfirmed"
        return {
            "phase": phase,
            "empty_mat_side": self.first_empty_mat_name,
            "button_press_status": button_status,
        }

    def _get_primary_failure_reason(self, success, terminal, first_press):
        if success:
            return "success"
        if self.press_cnt > 1:
            return "button_pressed_multiple_times"
        if self.press_cnt == 0:
            if not self._diagnostic_first_placement_ever_ready:
                return "first_placement_not_completed"
            if self._diagnostic_min_button_value < -0.001:
                return "button_press_insufficient"
            return "button_not_pressed"
        if self.stage_id == 0:
            first_press_conditions = first_press.get("conditions", {}) if first_press else {}
            if not first_press_conditions.get("block1_on_target_mat", False):
                return "pressed_before_block1_ready"
            if not first_press_conditions.get("block2_on_original_mat", False):
                return "pressed_after_block2_moved"
            if not (
                first_press_conditions.get("block1_below_height_limit", False)
                and first_press_conditions.get("block2_below_height_limit", False)
            ):
                return "pressed_while_block_held"
            return "invalid_first_press"

        terminal_conditions = terminal["conditions"]
        if terminal_conditions["button_latched_pressed"]:
            return "button_not_released"
        if not terminal_conditions["block1_on_target_mat"]:
            return "block1_disturbed_after_valid_press"
        if not terminal_conditions["block2_in_middle"]:
            return "block2_not_moved_to_middle"
        if not terminal_conditions["right_gripper_open"]:
            return "block2_not_released"
        return "final_conditions_not_confirmed"

    def get_eval_diagnostics(self, success):
        diagnostics = super().get_eval_diagnostics(success)
        terminal = self._get_rearrange_diagnostic_snapshot()
        events = getattr(self, "_eval_diagnostic_events", [])
        first_press = next(
            (
                event
                for event in events
                if event.get("name") == "button_pressed" and event.get("press_count") == 1
            ),
            None,
        )
        first_press_conditions = first_press.get("conditions", {}) if first_press else {}

        diagnostics["primary_failure_reason"] = self._get_primary_failure_reason(
            bool(success), terminal, first_press
        )
        diagnostics["conditions"] = {
            "first_placement_ever_ready": bool(self._diagnostic_first_placement_ever_ready),
            "button_pressed_at_least_once": self.press_cnt > 0,
            "button_pressed_exactly_once": self.press_cnt == 1,
            "valid_press_reached_stage_1": self.stage_id >= 1,
            "second_placement_ever_ready": bool(self._diagnostic_second_placement_ever_ready),
            "first_press_block1_on_target": first_press_conditions.get("block1_on_target_mat"),
            "first_press_block2_on_original": first_press_conditions.get("block2_on_original_mat"),
            "first_press_blocks_below_height_limit": (
                first_press_conditions.get("block1_below_height_limit", False)
                and first_press_conditions.get("block2_below_height_limit", False)
            ) if first_press else None,
            "terminal_block1_on_target": terminal["conditions"]["block1_on_target_mat"],
            "terminal_block2_in_middle": terminal["conditions"]["block2_in_middle"],
            "terminal_right_gripper_open": terminal["conditions"]["right_gripper_open"],
            "terminal_button_released": not terminal["conditions"]["button_latched_pressed"],
        }
        diagnostics["metrics"].update({
            "press_count": int(self.press_cnt),
            "stage_id": int(self.stage_id),
            "minimum_button_joint_position": float(self._diagnostic_min_button_value),
            **terminal["metrics"],
        })
        diagnostics["task_context"] = {
            "empty_mat_side": self.first_empty_mat_name,
            "initial_occupied_mat_side": self.initial_occupied_mat_side,
        }
        return diagnostics
