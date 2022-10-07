from skimage import io
from skimage.filters import threshold_otsu
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple


def load_image(
    path: Path, convert_color: bool = True, apply_where: bool = False
) -> np.array:
    if convert_color:
        return cv2.cvtColor(io.imread(path), cv2.COLOR_BGRA2GRAY)
    return np.where(io.imread(path) > 200, 255, 0) if apply_where else io.imread(path)


def save_image(img: np.array, path: Path) -> None:
    io.imsave(path, img)


def binarize_images(images_folder: Path, output_folder: Path) -> None:
    for f in images_folder.glob("*"):
        print(f)
        img = load_image(f)

        fname = f.stem
        _, binary_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        output_file = output_folder / (fname + ".jpg")
        save_image(binary_img, output_file)


def get_contour(images_folder: Path) -> None:
    for f in images_folder.glob("*"):
        if "_" in f.stem:
            continue
        print(f)
        img = load_image(f, False)

        fname = f.stem
        binary_img = cv2.Canny(img, 200, 255)

        output_file = images_folder / (fname + "_contour.jpg")
        save_image(binary_img, output_file)


def apply_salt_and_pepper(img: np.array) -> np.array:
    out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = np.random.uniform(0, 1)
            out[i, j] = 0 if rand < 0.02 else img[i, j] if rand < 0.98 else 255
    return out


def add_noise(images_folder: Path) -> None:
    for f in images_folder.glob("*"):
        if "_" in f.stem:
            continue
        print(f)
        img = load_image(f, False)
        print(img.shape)

        fname = f.stem
        binary_img = apply_salt_and_pepper(img)

        output_file = images_folder / (fname + "_noise.jpg")
        save_image(binary_img, output_file)


def load_pair_X_y(
    f: Path, images_folder: Path, target: str, apply_where: bool = False
) -> Tuple[np.array]:
    X_image = load_image(f, False, apply_where)
    y_image = load_image(
        images_folder / (f.stem + f"_{target}.jpg"), False, apply_where
    )
    return (X_image, y_image) if target != "noise" else (y_image, X_image)


def plot_image(img: np.array):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    binarize_images(Path("../images/"), Path("./images"))
    get_contour(Path("./images/"))
    add_noise(Path("./images/"))
