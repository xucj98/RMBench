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
