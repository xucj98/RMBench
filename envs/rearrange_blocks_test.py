from types import SimpleNamespace
from envs.rearrange_blocks import rearrange_blocks


def test_oracle_key_state_tracks_physical_task_progress():
    task = rearrange_blocks.__new__(rearrange_blocks)
    task._oracle_key_state_phase = 0
    task.stage_id = 0
    task.press_cnt = 0
    task.press_flag = False
    task.first_empty_mat_name = "left"
    task._oracle_left_arm_returned = lambda: True
    conditions = {"first_placement_ready": False, "right_gripper_open": False}
    task._get_rearrange_diagnostic_snapshot = lambda: {"conditions": conditions}

    assert task.get_oracle_key_state() == {
        "phase": "move_middle_block_to_empty_mat",
        "empty_mat_side": "left",
        "button_press_status": "NA",
    }

    conditions.update(first_placement_ready=True, right_gripper_open=True)
    assert task.get_oracle_key_state()["phase"] == "press_button_after_first_move"
    assert task.get_oracle_key_state()["button_press_status"] == "unconfirmed"

    task.stage_id = 1
    task.press_cnt = 1
    task.press_flag = True
    assert task.get_oracle_key_state()["button_press_status"] == "confirmed"

    task._oracle_left_arm_returned = lambda: False
    task.press_flag = False
    returning_state = task.get_oracle_key_state()
    assert returning_state["phase"] == "press_button_after_first_move"
    assert returning_state["button_press_status"] == "confirmed"

    task._oracle_left_arm_returned = lambda: True
    assert task.get_oracle_key_state() == {
        "phase": "move_original_mat_block_to_middle",
        "empty_mat_side": "left",
        "button_press_status": "NA",
    }


def test_oracle_left_arm_returned_accepts_pose_lists():
    task = rearrange_blocks.__new__(rearrange_blocks)
    task.get_arm_pose = lambda _arm: [0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    task.robot = SimpleNamespace(
        left_original_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )

    assert task._oracle_left_arm_returned()
