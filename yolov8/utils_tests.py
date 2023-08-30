import numpy as np
import utils


def test_numpy_bounding_boxes_conversions():
    expected_cxcywh = np.array([
        [1085.1736, 531.7921, 14.913665, 16.136751],
        [1193.0363, 532.3192, 54.93312, 17.184212],
        [1194.1448, 532.40845, 55.67838, 18.068424]
    ])

    expected_x1y1x2y2 = np.array([
       [1077.7168, 523.72375, 1092.6304, 539.8605],
       [1165.5697, 523.7271, 1220.5028, 540.9113],
       [1166.3055, 523.3742, 1221.984, 541.4427]
    ])

    actual_x1y1x2y2 = utils.xywh2xyxy(expected_cxcywh)
    actual_cxcywh = utils.xyxyxywh2(expected_x1y1x2y2)

    assert (np.round(actual_x1y1x2y2, 2) == np.round(expected_x1y1x2y2, 2)).all()
    assert (np.round(actual_cxcywh, 2) == np.round(expected_cxcywh, 2)).all()


if __name__ == "__main__":
    test_numpy_bounding_boxes_conversions()
    print("Everything passed")