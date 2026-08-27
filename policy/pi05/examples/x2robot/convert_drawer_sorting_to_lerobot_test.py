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


def test_completed_layers_follow_chronological_execution_order():
    timeline = converter.completed_layers_timeline(120, _record().intervals)

    assert timeline[29] == 0
    assert timeline[49] == 0
    assert timeline[50] == 1
    assert timeline[79] == 1
    assert timeline[80] == 2
    assert timeline[110] == 3


def test_execution_intervals_force_known_item_input_without_observation():
    input_ids, target_ids, target_mask, memory_action_valid = converter.memory_supervision_for_target_frames(
        _record(), target_frame_count=60, target_fps=15, query_stride=5
    )

    # 30 Hz raw frame 62 is inside label 5. The previous sampled state is
    # observe, but execution override makes this an item_2-conditioned action.
    np.testing.assert_array_equal(input_ids[31], [1, 2])
    np.testing.assert_array_equal(target_ids[31], [1, 2])
    assert memory_action_valid[31, 0]
    assert not memory_action_valid[31, 1]
    # Item 1 was genuinely acquired in label 1, so its execution memory remains
    # a valid dense action target.
    assert memory_action_valid[16].all()
    # Immediately after label 5 ends, the previous memory is item_2 and the
    # target is observe: this is the completion transition supervision.
    np.testing.assert_array_equal(input_ids[40], [1, 2])
    np.testing.assert_array_equal(target_ids[40], [2, 0])
    assert memory_action_valid[40].all()
    assert target_mask.all()


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


def test_shared_dense_encoding_concatenates_two_factorized_fields():
    encoded = converter.encode_shared_memory(np.asarray([[0, 0], [2, 3]]))

    np.testing.assert_array_equal(encoded[0], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(encoded[1], [0, 1, 0, 0, 0, 1])
