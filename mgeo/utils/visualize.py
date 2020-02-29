import matplotlib.pyplot as plt
import pathlib
from PIL import Image


def show(img, points=None, mark="b*", size=(6, 6), cmap="viridis", show_axis=False):
    plt.figure(figsize=size)
    if not show_axis:
        plt.axis("off")
    if points is not None and points.ndim == 2:
        if points.shape[1] == 2:
            plt.plot(points[:, 0], points[:, 1], mark)
        else:
            plt.plot(points[0], points[1], mark)
    plt.imshow(img, cmap=cmap)
    plt.show()


def show_imgs(img_list, title_list=None, rows=1, cmap="viridis", size=(16, 8), show_axis=False):
    if title_list is not None:
        assert len(img_list) == len(title_list)
    if type(img_list[0]) == pathlib.PosixPath or type(img_list[0]) == str:
        show_list = [Image.open(str(img_path)) for img_path in img_list]
    else:
        show_list = img_list
    plt.figure(figsize=size)
    assert len(show_list) % rows == 0
    for idx, img in enumerate(show_list):
        plt.subplot(rows, int(len(show_list) / rows), idx + 1)
        plt.imshow(img, cmap=cmap)
        if title_list is not None:
            plt.title(title_list[idx])
        if not show_axis:
            plt.axis("off")
    plt.show()


def show_histogram(img, size=(15, 6), bins=128, fill_contour=False, cmap="viridis", show_axis=False):
    assert img.ndim == 2
    plt.figure(figsize=size)

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap=cmap)
    plt.title("Original")
    plt.axis("equal")
    if not show_axis:
        plt.axis("off")
    mappable0 = plt.pcolormesh(img, cmap=cmap)
    plt.colorbar(mappable0, orientation="horizontal")

    # Contour
    plt.subplot(1, 3, 2)
    if fill_contour:
        plt.contourf(img, origin="image", cmap=cmap)
    else:
        plt.contour(img, origin="image", cmap=cmap)
    plt.title("Contour")
    plt.axis("equal")
    if not show_axis:
        plt.axis("off")
    plt.colorbar(mappable0, orientation="horizontal")

    # Histgram
    plt.subplot(1, 3, 3)
    plt.hist(img.flatten(), bins)
    plt.title("Histgram")
    plt.show()


def show_histograms(img_list, title_list=None, bins=256, cmap="viridis", size=(16, 8)):
    if title_list is not None:
        assert len(img_list) == len(title_list)
    plt.figure(figsize=size)
    for idx, img in enumerate(img_list):
        plt.subplot(1, len(img_list), idx + 1)
        plt.hist(img.flatten(), bins)
        if title_list is not None:
            plt.title(title_list[idx])
        plt.xlim(0, bins)
    plt.show()
