import pytest
import pandas as pd
import numpy as np
from pymlp.data.features import label_encode


@pytest.fixture
def data():
    return pd.DataFrame(
        data={
            "col1": ["Yes", "No", "Yes", "Maybe"],
            "col2": ["Today", "Today", "Tomorrow", "Yesterday"],
        }
    )


def test_label_encoder(data):
    expected: np.ndarray = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 2],
        ]
    )
    encoded: np.ndarray = label_encode(data)
    assert np.array_equal(expected, encoded)
