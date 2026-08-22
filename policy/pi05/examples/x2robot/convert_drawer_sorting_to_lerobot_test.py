from pathlib import Path

import convert_drawer_sorting_to_lerobot as converter
import numpy as np


def _record() -> converter.EpisodeRecord:
    return converter.EpisodeRecord(
        path=Path("unused"),
        total_frames=120,
        source_fps=30.0,
        intervals=(
            converter.Interval(10, 20, "1", 1, "observe"),
            converter.Interval(30, 50, "4", 1, "execute"),
            converter.Interval(60, 80, "5", 2, "execute"),
            converter.Interval(90, 110, "6", 3, "execute"),
        ),
        missing_observation_items=(2, 3),
        annotation_path=Path("anno/sort.json"),
    )


def test_memory_timeline_acquires_holds_and_returns_to_observe():
    timeline = converter.semantic_memory_timeline(120, _record().intervals)

    assert timeline[9] == 0
    assert timeline[10] == 1
    assert timeline[29] == 1
    assert timeline[30] == 1
    assert timeline[49] == 1
    assert timeline[50] == 0
    assert timeline[60] == 2
    assert timeline[80] == 0


def test_execution_intervals_force_known_item_input_without_observation():
    input_ids, target_ids, target_mask, memory_action_valid = converter.memory_supervision_for_target_frames(
        _record(), target_frame_count=60, target_fps=15, query_stride=5
    )

    # 30 Hz raw frame 62 is inside label 5. The previous sampled state is
    # observe, but execution override makes this an item_2-conditioned action.
    assert input_ids[31, 0] == 2
    assert target_ids[31, 0] == 2
    assert not memory_action_valid[31, 0]
    # Item 1 was genuinely acquired in label 1, so its execution memory remains
    # a valid dense action target.
    assert memory_action_valid[16, 0]
    # Immediately after label 5 ends, the previous memory is item_2 and the
    # target is observe: this is the completion transition supervision.
    assert input_ids[40, 0] == 2
    assert target_ids[40, 0] == 0
    assert memory_action_valid[40, 0]
    assert target_mask[:, 0].all()


def test_dense_encoding_reserves_unknown_as_all_zero():
    encoded = converter.implicit_unknown_one_hot(np.asarray([0, 1, 2, 3]))

    np.testing.assert_array_equal(
        encoded,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
