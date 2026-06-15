from ._base_task import Base_Task
from .utils import *

class battery_try(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.battery_slot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="017_battery_slot_gauge",
            modelid=10127,
            xlim=[0.0, 0.0],
            ylim=[-0.1, -0.1],
            rotate_rand=False,
            qpos=[0.707, 0 , 0, 0.707],
            fix_root_link=True,
        )
        self.set_dashboard_off()

        self.battery1 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="018_battery",
            modelid=10128,
            xlim=[-0.2, -0.25],
            ylim=[-0.2, -0.1],
            zlim=[0.761, 0.761],
            rotate_rand=False,
            qpos=[0.707, -0.707, 0, 0],
            fix_root_link=False,
        )
        self.battery1.set_mass(0.01)
        self.battery2 = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="018_battery",
            modelid=10128,
            xlim=[0.2, 0.25],
            ylim=[-0.2, -0.1],
            zlim=[0.761, 0.761],
            rotate_rand=False,
            qpos=[0.707, -0.707, 0, 0],
            fix_root_link=False,
        )
        self.battery2.set_mass(0.01)
        self.target_pose1_p = self.battery_slot.get_pose().p + np.array([-0.02, 0.0, 0.0])
        self.target_pose2_p = self.battery_slot.get_pose().p + np.array([0.02, 0.0, 0.0])   
        self.quat_of_target_pose_0 = np.array([0.707, -0.707, 0.0, 0.0], dtype=np.float64)
        self.quat_of_target_pose_1 = np.array([0.0, 0.0, 0.707, -0.707], dtype=np.float64)
        self.combination_lst = [[0,0], [1,0], [1,1], [0,1]]
        self.correct_combination = self.combination_lst[np.random.randint(1, len(self.combination_lst))]

    def _combination_name(self, combination):
        return "".join(str(int(value)) for value in combination)
    
    def set_dashboard_off(self, joint_names=['needle_joint'], target=0.0): 
        art = self.battery_slot.actor if hasattr(self.battery_slot, "actor") else self.battery_slot   
        joints = art.get_active_joints()   
        joint_names_list = [j.get_name() for j in joints]    
        for joint_name in joint_names:
            idx = joint_names_list.index(joint_name)
            qpos = art.get_qpos()
            qpos[idx] = target   
            art.set_qpos(qpos) 
            joints[idx].set_drive_target(target)

    def set_dashboard_on(self, joint_names=['needle_joint'], target=1.57): 
        art = self.battery_slot.actor if hasattr(self.battery_slot, "actor") else self.battery_slot   
        joints = art.get_active_joints()   
        joint_names_list = [j.get_name() for j in joints]    
        for joint_name in joint_names:
            idx = joint_names_list.index(joint_name)
            qpos = art.get_qpos()
            qpos[idx] = target   
            art.set_qpos(qpos) 
            joints[idx].set_drive_target(target)

    def check_dashboard_on(self):
        art = self.battery_slot.actor if hasattr(self.battery_slot, "actor") else self.battery_slot  
        joints = art.get_active_joints()   
        joint_names_list = [j.get_name() for j in joints]  
        idx =  joint_names_list.index('needle_joint')
        qpos = art.get_qpos()
        if qpos[idx] > 1.4:
            return True
        else:
            return False

    def get_battery_state(self, battery_actor):
        curr_q = battery_actor.get_pose().q
        sim0 = abs(float(np.dot(curr_q, self.quat_of_target_pose_0))) 
        sim180 = abs(float(np.dot(curr_q, self.quat_of_target_pose_1))) 
        state = 0 if sim0 >= sim180 else 1
        return state
        
    def check_battery_in_slot(self, battery_actor, target_pose_p):
        curr_p = battery_actor.get_pose().p
        battery_in_slot = (np.linalg.norm(curr_p[:2] - target_pose_p[:2])<0.02)
        return battery_in_slot
    
    def get_curr_combination(self):
        battery1_state = self.get_battery_state(self.battery1)
        battery2_state = self.get_battery_state(self.battery2)
        curr_combination = [battery1_state, battery2_state]
        return curr_combination

    def _move_stage(self, stage_name, *actions, language_annotation):
        start = self._key_state_stage_start()
        self.move(*actions, language_annotation=language_annotation)
        self._record_key_state_micro_stage(stage_name, start)

    def place_for_battery1(self, arm, state, stage_prefix):
        curr_state = self.get_battery_state(self.battery1)
        if state == curr_state:
            return False
        target_quat = self.quat_of_target_pose_0 if state == 0 else self.quat_of_target_pose_1
        direction = "positive" if state == 0 else "negative"
        annotation = f"Pick up the left battery and place it into the battery slot in the {direction} direction."
        self._move_stage(
            f"{stage_prefix}_battery1_pick",
            self.grasp_actor(self.battery1, arm_tag=arm, pre_grasp_dis=0.1, grasp_dis=0.03),
            language_annotation=annotation,
        )
        self._move_stage(
            f"{stage_prefix}_battery1_lift",
            self.move_by_displacement(arm_tag=arm, z=0.05),
            language_annotation=annotation,
        )
        self._move_stage(
            f"{stage_prefix}_battery1_place",
            self.place_actor(
                self.battery1,
                arm_tag=arm,
                target_pose=self.target_pose1_p.tolist()+target_quat.tolist(),
                functional_point_id=0,
                constrain="align",
            ),
            language_annotation=annotation,
        )
        self.check_correct_combination()
        self._move_stage(
            f"{stage_prefix}_battery1_return",
            self.back_to_origin(arm_tag=arm),
            language_annotation=annotation,
        )
        return True

    def place_for_battery2(self, arm, state, stage_prefix):
        curr_state = self.get_battery_state(self.battery2)
        if state == curr_state:
            return False
        target_quat = self.quat_of_target_pose_0 if state == 0 else self.quat_of_target_pose_1
        direction = "positive" if state == 0 else "negative"
        annotation = f"Pick up the right battery and place it into the battery slot in the {direction} direction."
        self._move_stage(
            f"{stage_prefix}_battery2_pick",
            self.grasp_actor(self.battery2, arm_tag=arm, pre_grasp_dis=0.1, grasp_dis=0.03),
            language_annotation=annotation,
        )
        self._move_stage(
            f"{stage_prefix}_battery2_lift",
            self.move_by_displacement(arm_tag=arm, z=0.05),
            language_annotation=annotation,
        )
        self._move_stage(
            f"{stage_prefix}_battery2_place",
            self.place_actor(
                self.battery2,
                arm_tag=arm,
                target_pose=self.target_pose2_p.tolist()+target_quat.tolist(),
                functional_point_id=0,
                constrain="align",
            ),
            language_annotation=annotation,
        )
        self.check_correct_combination()
        self._move_stage(
            f"{stage_prefix}_battery2_return",
            self.back_to_origin(arm_tag=arm),
            language_annotation=annotation,
        )
        return True

    def check_correct_combination(self):
        curr_combination = self.get_curr_combination()
        if curr_combination == self.correct_combination:
            self.set_dashboard_on()
 
    def play_once(self):
        actual_trial_sequence = []
        dashboard_on_after_trial = None
        for idx, next_comb in enumerate(self.combination_lst):
            stage_name = "place_initial_00" if idx == 0 else f"try_{self._combination_name(next_comb)}"
            arm_tag = ArmTag("left" if self.battery1.get_pose().p[0] < 0 else "right")
            if idx == 0:
                self._move_stage(
                    f"{stage_name}_dual_pick",
                    self.grasp_actor(self.battery1, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.03),
                    self.grasp_actor(self.battery2, arm_tag=arm_tag.opposite, pre_grasp_dis=0.1, grasp_dis=0.03),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self._move_stage(
                    f"{stage_name}_dual_lift",
                    self.move_by_displacement(arm_tag=arm_tag, z=0.1),
                    self.move_by_displacement(arm_tag=arm_tag.opposite, z=0.1),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self._move_stage(
                    f"{stage_name}_battery1_place",
                    self.place_actor(self.battery1, arm_tag=arm_tag, target_pose=self.target_pose1_p.tolist()+self.quat_of_target_pose_0.tolist(), functional_point_id=0, constrain="align"),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self._move_stage(
                    f"{stage_name}_battery1_return",
                    self.back_to_origin(arm_tag=arm_tag),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self._move_stage(
                    f"{stage_name}_battery2_place",
                    self.place_actor(self.battery2, arm_tag=arm_tag.opposite, target_pose=self.target_pose2_p.tolist()+self.quat_of_target_pose_0.tolist(), functional_point_id=0, constrain="align"),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
                self._move_stage(
                    f"{stage_name}_battery2_return",
                    self.back_to_origin(arm_tag=arm_tag.opposite),
                    language_annotation="Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.")
            else:
                self.place_for_battery1(arm_tag, next_comb[0], stage_name)
                self.place_for_battery2(arm_tag.opposite, next_comb[1], stage_name)
            curr_state = self.get_curr_combination()
            actual_trial_sequence.append(self._combination_name(curr_state))
            if curr_state == self.correct_combination:
                dashboard_on_after_trial = self._combination_name(curr_state)
                break
        self._set_key_state_scene_info(
            task_facts={
                "correct_combination": self._combination_name(self.correct_combination),
                "trial_order": [self._combination_name(item) for item in self.combination_lst],
                "actual_trial_sequence": actual_trial_sequence,
                "dashboard_on_after_trial": dashboard_on_after_trial,
            },
            phase_sequence=[
                "place_initial_00",
                "try_10",
                "try_11",
                "try_01",
            ],
        )
        return self.info

    def check_success(self):
        current_combination = self.get_curr_combination()
        if current_combination == self.correct_combination:
            self.set_dashboard_on()
        if self.check_battery_in_slot(self.battery1, self.target_pose1_p) and self.check_battery_in_slot(self.battery2, self.target_pose2_p) \
            and current_combination == self.correct_combination and self.check_dashboard_on():
            return True
        else:
            return False
