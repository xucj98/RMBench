from pathlib import Path

import convert_robotwin_key_state_to_lerobot as converter
import numpy as np
import pytest


def test_declarative_structured_state_tokens_resolve_phase_and_attribute():
    structured = {
        "fields": [
            {
                "name": "phase",
                "labels": ["p0", "p1"],
                "ranges": [
                    {"label": "p0", "window": ["episode_start", 4]},
                    {"label": "p1", "window": [4, "episode_end"]},
                ],
            },
            {
                "name": "origin",
                "labels": ["unknown", "left", "right"],
                "transitions": [
                    {
                        "from_value": "unknown",
                        "to_value": "right",
                        "update_window": ["episode_start", 3],
                    }
                ],
            },
        ]
    }
    resolvers = converter._structured_state_token_resolvers(structured, {}, 10)  # noqa: SLF001

    np.testing.assert_array_equal(
        converter._structured_state_token_ids(resolvers, {}, 10, 0),  # noqa: SLF001
        [0, 2],
    )
    np.testing.assert_array_equal(
        converter._structured_state_token_ids(resolvers, {}, 10, 7),  # noqa: SLF001
        [1, 2],
    )


def test_partial_declarative_state_token_schema_fails_fast():
    structured = {
        "fields": [
            {"name": "phase", "labels": ["p0", "p1"], "ranges": [{"label": "p0", "window": [0, 1]}]},
            {"name": "attribute", "labels": ["unknown", "known"]},
        ]
    }

    with pytest.raises(ValueError, match="non-declarative"):
        converter._structured_state_token_resolvers(structured, {}, 2)  # noqa: SLF001


def test_rearrange_adapters_resolve_the_same_semantic_memory():
    project_root = Path(__file__).resolve().parents[4]
    config_path = project_root / "converter_configs/shared_memory/rearrange_blocks.yaml"
    config = converter._merge_memory_schema(converter._load_yaml(config_path), config_path)  # noqa: SLF001

    assert config["semantic_memory"]["fields"] == config["structured_state_tokens"]["fields"]
    assert [field["name"] for field in config["structured_state_tokens"]["fields"]] == [
        "phase",
        "empty_mat_side",
        "button_press_status",
    ]
    assert config["phase"]["dim"] == [14, 17]
    assert config["attributes"][0]["dim"] == [17, 20]
    assert config["execution"][0]["dim"] == [20, 23]


@pytest.mark.parametrize(
    ("task", "field_names", "dense_dims"),
    [
        ("put_back_block", ["phase", "origin_mat"], [[14, 17], [17, 22]]),
        (
            "swap_blocks",
            ["phase", "initial_empty_tray", "first_origin_tray"],
            [[14, 18], [18, 22], [22, 26]],
        ),
        ("battery_try", ["phase"], [[14, 18]]),
        (
            "cover_blocks",
            ["phase", "red_pos", "green_pos", "blue_pos"],
            [[14, 20], [20, 24], [24, 28], [28, 32]],
        ),
    ],
)
def test_multitask_shared_adapters_resolve_the_same_semantic_memory(task, field_names, dense_dims):
    project_root = Path(__file__).resolve().parents[4]
    config_path = project_root / f"converter_configs/shared_memory/{task}.yaml"
    config = converter._merge_memory_schema(converter._load_yaml(config_path), config_path)  # noqa: SLF001

    assert config["semantic_memory"]["fields"] == config["structured_state_tokens"]["fields"]
    assert [field["name"] for field in config["semantic_memory"]["fields"]] == field_names
    resolved_dense_fields = [config["phase"], *config.get("attributes", []), *config.get("execution", [])]
    assert [field["dim"] for field in resolved_dense_fields] == dense_dims
    assert config["structured_state_tokens"]["query_stride"] == 20


def test_rearrange_button_timeline_uses_schema_language_event():
    project_root = Path(__file__).resolve().parents[4]
    token_path = project_root / "converter_configs/shared_memory/rearrange_blocks.yaml"
    config = converter._merge_memory_schema(converter._load_yaml(token_path), token_path)  # noqa: SLF001
    button = config["structured_state_tokens"]["fields"][2]
    info = {
        "micro_stages": [
            {"name": "block1_place", "start_frame": 10, "end_frame": 20},
            {"name": "press_return", "start_frame": 70, "end_frame": 80},
        ],
        "_language_segments": [["segment", 10] for _ in range(11)],
    }
    ranges = converter._range_labels(button, info, 110, "button_press_status")  # noqa: SLF001

    assert converter._label_at(10, ranges) == 0  # noqa: SLF001
    assert converter._label_at(30, ranges) == 1  # noqa: SLF001
    assert converter._label_at(50, ranges) == 2  # noqa: SLF001
    assert converter._label_at(90, ranges) == 0  # noqa: SLF001


def test_state_token_training_pair_uses_previous_query_target():
    def resolve_ids(index):
        return np.asarray([index // 20, 2], dtype=np.int64)

    initial_ids = np.asarray([0, 0], dtype=np.int64)

    input_ids, target_ids = converter._state_token_training_pair(resolve_ids, 10, 20, initial_ids)  # noqa: SLF001
    np.testing.assert_array_equal(input_ids, [0, 0])
    np.testing.assert_array_equal(target_ids, [0, 2])
