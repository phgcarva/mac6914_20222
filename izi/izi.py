from misc import load_image, load_pair_X_y, plot_image
from pathlib import Path
import numpy as np
from typing import Dict, Any


def build_model_izi(
    images_folder: Path, window_is_cross: bool = True, target: str = "contour"
):
    freq_table = {}
    for f in images_folder.glob("*"):
        if f.stem.isnumeric():
            print(f)
            X, y = load_pair_X_y(f, images_folder, target, True)
            # plot_image(y)
            freq_table = update_frequecy_table(
                X,
                y,
                window_is_cross,
                freq_table,
            )
    truth_table = make_truth_table(freq_table)
    return (
        make_model_cross(truth_table)
        if window_is_cross
        else make_model_square(truth_table)
    )


def update_frequecy_table(
    img: np.array,
    target: np.array,
    window_is_cross: bool,
    table: Dict[Any, Any],
) -> Dict[Any, Any]:
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            key = (
                get_cross_on_image(img, row, col)
                if window_is_cross is True
                else get_square_on_image(img, row, col),
            )
            target_val = target[row][col] / 255
            if key not in table:
                table.update({key: {0: 0, 1: 0}})
            table[key][target_val] += 1
    return table


def make_truth_table(frequencies: Dict[Any, Any]) -> Dict:
    truth_table = [
        (key, 0 if frequencies[key][0] > frequencies[key][1] else 1)
        for key in sorted(frequencies.keys())
    ]

    return dict(truth_table)


def get_positive_keys(truth_table: Dict):
    positive_keys = [key[0] for key in truth_table if truth_table[key] == 1]

    return positive_keys


def make_model_cross(truth_table: Dict):
    positive_keys = get_positive_keys(truth_table)

    def model(x1: bool, x2: bool, x3: bool, x4: bool, x5: bool):
        formula = False
        for key in positive_keys:
            factor = (
                (x1 if key[0] == "1" else not x1)
                and (x2 if key[1] == "1" else not x1)
                and (x3 if key[2] == "1" else not x1)
                and (x4 if key[3] == "1" else not x1)
                and (x5 if key[4] == "1" else not x1)
            )
            formula = formula or factor
        return 0 if not formula else 255

    return model


def make_model_square(truth_table: Dict):
    positive_keys = get_positive_keys(truth_table)

    def model(
        x1: bool,
        x2: bool,
        x3: bool,
        x4: bool,
        x5: bool,
        x6: bool,
        x7: bool,
        x8: bool,
        x9: bool,
    ):
        formula = False
        for key in positive_keys:
            factor = (
                (x1 if key[0] == "1" else not x1)
                and (x2 if key[1] == "1" else not x1)
                and (x3 if key[2] == "1" else not x1)
                and (x4 if key[3] == "1" else not x1)
                and (x5 if key[4] == "1" else not x1)
                and (x6 if key[5] == "1" else not x1)
                and (x7 if key[6] == "1" else not x1)
                and (x8 if key[7] == "1" else not x1)
                and (x9 if key[8] == "1" else not x1)
            )
            formula = formula or factor
        return 0 if not formula else 255

    return model


def get_cross_on_image(
    img: np.array,
    row: int,
    col: int,
    return_str: bool = True,
):
    x1 = int(0 if row == 0 else img[row - 1][col] / 255)
    x2 = int(0 if col == 0 else img[row][col - 1] / 255)
    x3 = int(img[row][col] / 255)
    x4 = int(0 if col == img.shape[1] - 1 else img[row][col + 1] / 255)
    x5 = int(0 if row == img.shape[0] - 1 else img[row + 1][col] / 255)
    return (
        get_cross_tablekey(x1, x2, x3, x4, x5)
        if return_str
        else (bool(x1), bool(x2), bool(x3), bool(x4), bool(x5))
    )


def get_square_on_image(
    img: np.array,
    row: int,
    col: int,
    return_str: bool = True,
):
    x1 = int(0 if (row == 0 or col == 0) else img[row - 1][col - 1] / 255)
    x2 = int(0 if row == 0 else img[row - 1][col] / 255)
    x3 = int(
        0 if (row == 0 or col == img.shape[1] - 1) else img[row - 1][col + 1] / 255
    )
    x4 = int(0 if col == 0 else img[row][col - 1] / 255)
    x5 = int(img[row][col] / 255)
    x6 = int(0 if col == img.shape[1] - 1 else img[row][col + 1] / 255)
    x7 = int(
        0 if (row == img.shape[0] - 1 or col == 0) else img[row + 1][col - 1] / 255
    )
    x8 = int(0 if row == img.shape[0] - 1 else img[row + 1][col] / 255)
    x9 = int(
        0
        if (row == img.shape[0] - 1 or col == img.shape[1] - 1)
        else img[row - 1][col + 1] / 255
    )
    return (
        get_square_tablekey(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        if return_str
        else (
            bool(x1),
            bool(x2),
            bool(x3),
            bool(x4),
            bool(x5),
            bool(x6),
            bool(x7),
            bool(x8),
            bool(x9),
        )
    )


def get_cross_tablekey(x1: int, x2: int, x3: int, x4: int, x5: int):
    return str(x1) + str(x2) + str(x3) + str(x4) + str(x5)


def get_square_tablekey(
    x1: int,
    x2: int,
    x3: int,
    x4: int,
    x5: int,
    x6: int,
    x7: int,
    x8: int,
    x9: int,
):
    return (
        str(x1)
        + str(x2)
        + str(x3)
        + str(x4)
        + str(x5)
        + str(x6)
        + str(x7)
        + str(x8)
        + str(x9)
    )


if __name__ == "__main__":
    target = "contour"
    window_is_cross = True

    model = build_model_izi(
        Path("./images"),
        target=target,
        window_is_cross=window_is_cross,
    )
    img = load_image(Path("./images/23.jpg"), False, True)
    pred = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if window_is_cross:
                x1, x2, x3, x4, x5 = get_cross_on_image(img, row, col, False)
                pred[row, col] = model(x1, x2, x3, x4, x5)
            else:
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = get_square_on_image(
                    img, row, col, False
                )
                pred[row, col] = model(x1, x2, x3, x4, x5, x6, x7, x8, x9)

    plot_image(pred)
