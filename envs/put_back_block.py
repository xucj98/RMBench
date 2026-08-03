from ._base_task import Base_Task
from .utils import *

class put_back_block(Base_Task):
    
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.button = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="005_button",
            modelid=10124,
            xlim=[-0.25, -0.25],
            ylim=[-0.1, -0.1],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 0],
            fix_root_link=True,
        )
        self.button.set_mass(0.0001,['button_cap'])
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
        
        mat_half_size = [0.04, 0.04, 0.0005]
        mats_pose = []
        x_mat = 0.0
        for _ in range(2):
            mat_pos = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[-0.1, -0.1],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            x_mat += 0.2
        y_mat = -0.2
        for _ in range(2):
            mat_pos = rand_pose(
                xlim=[0.1, 0.1],
                ylim=[y_mat, y_mat],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            y_mat += 0.2
        self.mat_lst=[]
        for i in range(4):
            mat = create_mat(mats_pose[i])
            self.mat_lst.append(mat)
        self.block_half_size = 0.02
        block_id = np.random.randint(0,4)
        self.mat_name = ['left', 'right', 'front', 'back'][block_id]
        self.block = create_block(mats_pose[block_id])
        self.target_pose = self.mat_lst[block_id].get_pose().p
        self.center_pose = [0.1,-0.1,0.765,1,0,0,0]
        self.stage_id = 0
        self._diagnostic_center_ever_ready = False
        self._diagnostic_origin_return_ever_ready = False
        self._diagnostic_min_button_value = 0.0

    @staticmethod
    def _vec_to_list(vec):
        return [float(x) for x in np.asarray(vec).reshape(-1)]

    def _pose_to_dict(self, pose):
        return {
            "p": self._vec_to_list(pose.p),
            "q": self._vec_to_list(pose.q),
        }

    def _build_task_facts(self):
        mat_names = ['left', 'right', 'front', 'back']
        return {
            "origin_mat_id": int(mat_names.index(self.mat_name)),
            "origin_mat_name": self.mat_name,
            "block_half_size": float(self.block_half_size),
            "initial_block_pose": self._pose_to_dict(self.block.get_pose()),
            "target_pose": self._vec_to_list(self.target_pose),
            "center_pose": self._vec_to_list(self.center_pose),
            "mat_poses": {
                name: self._pose_to_dict(mat.get_pose())
                for name, mat in zip(mat_names, self.mat_lst)
            },
            "button_pose": self._pose_to_dict(self.button.get_pose()),
        }
    
    def play_once(self):
        task_facts = self._build_task_facts()
        start = self._key_state_stage_start()
        self.move(self.grasp_actor(self.block,arm_tag="right",pre_grasp_dis=0.1, grasp_dis=0.02), language_annotation=f'Pick up the block and move it to the center position.')
        self._record_key_state_micro_stage("center_pick", start)
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Pick up the block and move it to the center position.')
        self._record_key_state_micro_stage("center_lift", start)
        start = self._key_state_stage_start()
        self.move(self.place_actor(self.block, arm_tag="right", target_pose=self.center_pose, functional_point_id=2, dis=0.02), language_annotation=f'Pick up the block and move it to the center position.')
        self._record_key_state_micro_stage("center_place", start)
        self.check_block_in_center()
        self.press_button()
        start = self._key_state_stage_start()
        self.move(self.back_to_origin(arm_tag="left"), language_annotation=f'Press the button.')
        self._record_key_state_micro_stage("button_return", start)
        start = self._key_state_stage_start()
        self.move(self.grasp_actor(self.block,arm_tag="right",pre_grasp_dis=0.02, grasp_dis=0.02), language_annotation=f'Move the block back to the {self.mat_name} mat.')
        self._record_key_state_micro_stage("origin_pick", start)
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="right", z=0.1), language_annotation=f'Move the block back to the {self.mat_name} mat.')
        self._record_key_state_micro_stage("origin_lift", start)
        start = self._key_state_stage_start()
        self.move(self.place_actor(self.block, arm_tag="right", target_pose=self.target_pose, functional_point_id=2, dis=0.01), language_annotation=f'Move the block back to the {self.mat_name} mat.')
        self._record_key_state_micro_stage("origin_place", start)
        task_facts["final_block_pose"] = self._pose_to_dict(self.block.get_pose())
        self._set_key_state_scene_info(
            task_facts=task_facts,
            phase_sequence=[
                "move_block_to_center",
                "press_button",
                "move_block_back_to_origin_mat",
            ],
        )
        return self.info
        
    def press_button(self):
        start = self._key_state_stage_start()
        self.move(self.grasp_actor(self.button, arm_tag="left", pre_grasp_dis=0.08, grasp_dis=0.08, contact_point_id=0), language_annotation=f'Press the button.')
        self._record_key_state_micro_stage("button_approach", start)
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="left", z=-0.04), language_annotation=f'Press the button.')
        self._record_key_state_micro_stage("button_press", start)
        self.update_press_success()
        self.check_success()
        start = self._key_state_stage_start()
        self.move(self.move_by_displacement(arm_tag="left", z=0.04), language_annotation=f'Press the button.')
        self._record_key_state_micro_stage("button_release", start)
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
    
    def update_press_success(self):
        button_value = float(self.get_current_button_value("button"))
        self._diagnostic_min_button_value = min(
            self._diagnostic_min_button_value,
            button_value,
        )
        if button_value < -0.005 and not self.press_flag:
            self.press_flag = True
            self.press_cnt += 1
            snapshot = self._get_put_back_diagnostic_snapshot()
            self._record_eval_diagnostic_event(
                "button_pressed",
                press_count=self.press_cnt,
                stage_id=self.stage_id,
                conditions=snapshot["conditions"],
                metrics=snapshot["metrics"],
            )

    def check_block_in_center(self):
        block_pose = self.block.get_pose().p
        return np.abs(block_pose[0] - self.center_pose[0]) < 0.04 and np.abs(block_pose[1] - self.center_pose[1]) < 0.04 and block_pose[2] < 0.77

    def _get_put_back_diagnostic_snapshot(self):
        block_pose = self.block.get_pose().p
        center_dx = float(np.abs(block_pose[0] - self.center_pose[0]))
        center_dy = float(np.abs(block_pose[1] - self.center_pose[1]))
        origin_dx = float(np.abs(block_pose[0] - self.target_pose[0]))
        origin_dy = float(np.abs(block_pose[1] - self.target_pose[1]))
        button_value = float(self.get_current_button_value("button"))

        conditions = {
            "block_in_center_xy": center_dx < 0.04 and center_dy < 0.04,
            "block_below_height_limit": bool(block_pose[2] < 0.77),
            "block_on_origin_mat_xy": origin_dx < 0.03 and origin_dy < 0.03,
            "right_gripper_open": bool(self.is_right_gripper_open()),
            "button_latched_pressed": bool(self.press_flag),
        }
        conditions["center_ready"] = bool(
            conditions["block_in_center_xy"]
            and conditions["block_below_height_limit"]
        )
        conditions["origin_return_ready"] = bool(
            conditions["block_on_origin_mat_xy"]
            and conditions["block_below_height_limit"]
            and conditions["right_gripper_open"]
            and not conditions["button_latched_pressed"]
        )
        return {
            "conditions": conditions,
            "metrics": {
                "button_joint_position": button_value,
                "block_center_dx": center_dx,
                "block_center_dy": center_dy,
                "block_origin_dx": origin_dx,
                "block_origin_dy": origin_dy,
                "block_z": float(block_pose[2]),
            },
        }

    def _update_put_back_diagnostic_progress(self, snapshot):
        conditions = snapshot["conditions"]
        if conditions["center_ready"] and not self._diagnostic_center_ever_ready:
            self._diagnostic_center_ever_ready = True
            self._record_eval_diagnostic_event(
                "center_ready",
                stage_id=self.stage_id,
                conditions=conditions,
                metrics=snapshot["metrics"],
            )
        if (
            self.stage_id >= 1
            and conditions["origin_return_ready"]
            and not self._diagnostic_origin_return_ever_ready
        ):
            self._diagnostic_origin_return_ever_ready = True
            self._record_eval_diagnostic_event(
                "origin_return_ready",
                stage_id=self.stage_id,
                conditions=conditions,
                metrics=snapshot["metrics"],
            )

    def check_success(self):
        if self.stage_id == 2:
            self.max_reward = max(self.max_reward, 1.0)
            return True
        self.update_button_reset(self.button)
        self.update_press_success()
        self.set_button_unpressed(self.button, target=min(0.0, self.get_current_button_value("button")+0.002))

        snapshot = self._get_put_back_diagnostic_snapshot()
        self._update_put_back_diagnostic_progress(snapshot)
        conditions = snapshot["conditions"]

        if self.press_flag:
            if self.stage_id == 0 and self.press_cnt == 1 and conditions["center_ready"]:
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
            if self.stage_id == 1 and conditions["origin_return_ready"]:
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

    def _get_primary_failure_reason(self, success, terminal, first_press):
        if success:
            return "success"
        if self.stage_id == 0:
            if self.press_cnt == 0:
                if not self._diagnostic_center_ever_ready:
                    return "block_not_moved_to_center"
                if self._diagnostic_min_button_value < -0.001:
                    return "button_press_insufficient"
                return "button_not_pressed_after_center"

            first_press_conditions = first_press.get("conditions", {}) if first_press else {}
            if not first_press_conditions.get("block_in_center_xy", False):
                return "pressed_before_block_centered"
            if not first_press_conditions.get("block_below_height_limit", False):
                return "pressed_while_block_held"
            if self.press_cnt > 1:
                return "button_pressed_multiple_times_after_invalid_press"
            return "invalid_first_press"

        terminal_conditions = terminal["conditions"]
        if terminal_conditions["button_latched_pressed"]:
            return "button_not_released"
        if not terminal_conditions["block_on_origin_mat_xy"]:
            return "block_not_returned_to_origin_mat"
        if not terminal_conditions["block_below_height_limit"]:
            return "block_held_above_origin_mat"
        if not terminal_conditions["right_gripper_open"]:
            return "block_not_released_at_origin_mat"
        return "final_conditions_not_confirmed"

    def get_eval_diagnostics(self, success):
        diagnostics = super().get_eval_diagnostics(success)
        terminal = self._get_put_back_diagnostic_snapshot()
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
            "center_ever_ready": bool(self._diagnostic_center_ever_ready),
            "button_pressed_at_least_once": self.press_cnt > 0,
            "button_pressed_exactly_once": self.press_cnt == 1,
            "valid_press_reached_stage_1": self.stage_id >= 1,
            "origin_return_ever_ready": bool(self._diagnostic_origin_return_ever_ready),
            "first_press_block_in_center": first_press_conditions.get("block_in_center_xy"),
            "first_press_block_below_height_limit": first_press_conditions.get("block_below_height_limit"),
            "terminal_block_on_origin_mat": terminal["conditions"]["block_on_origin_mat_xy"],
            "terminal_block_below_height_limit": terminal["conditions"]["block_below_height_limit"],
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
            "origin_mat": self.mat_name,
        }
        return diagnostics
