"""Small checks for the disabled shared-cache experiment."""

import pytest

from tools.cache_experiment import shared_token_prefix


def test_shared_token_prefix():
    assert shared_token_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == [1, 2]


@pytest.mark.parametrize("sequences", [[], [[1], []]])
def test_shared_token_prefix_rejects_empty_sequences(sequences):
    with pytest.raises(ValueError):
        shared_token_prefix(sequences)
