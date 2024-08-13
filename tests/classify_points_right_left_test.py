import numpy as np

from fibsight.texture.gabor import classify_points_relative_to_line


def test_classify_points_relative_to_line():
    # Test horizontal line
    line_start = (0, 0)
    line_end = (10, 0)
    points = np.array(
        [
            [5, 1],  # Above the line (right)
            [5, -1],  # Below the line (left)
            [5, 0],  # On the line
        ]
    )
    expected = np.array([1, -1, 1])
    assert np.array_equal(
        classify_points_relative_to_line(points, line_start, line_end), expected
    )

    # Test diagonal line (45 degrees)
    line_start = (0, 0)
    line_end = (10, 10)
    points = np.array(
        [
            [5, 6],  # Right of the line
            [6, 5],  # Left of the line
            [5, 5],  # On the line
        ]
    )
    expected = np.array([1, -1, 1])
    assert np.array_equal(
        classify_points_relative_to_line(points, line_start, line_end), expected
    )

    # Test points with various shapes
    points = np.array(
        [
            # 2D array (2 points, both right and left of the line)
            [[2, 3], [4, 2]],
            # 2D array (2 points, both on and left of the line)
            [[2, 2], [0, 0]],
        ]
    )
    expected = np.array([[1, -1], [1, 1]])
    assert np.array_equal(
        classify_points_relative_to_line(points, line_start, line_end), expected
    )
