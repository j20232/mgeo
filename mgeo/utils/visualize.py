import matplotlib.pyplot as plt


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


def show_imgs(img_list, title_list=None, cmap="viridis", size=(16, 8), show_axis=False):
    if title_list is not None:
        assert len(img_list) == len(title_list)
    plt.figure(figsize=size)
    for idx, img in enumerate(img_list):
        plt.subplot(1, len(img_list), idx + 1)
        plt.imshow(img, cmap=cmap)
        if title_list is not None:
            plt.title(title_list[idx])
        if not show_axis:
            plt.axis("off")
    plt.show()


def show_histgram(img, size=(15, 6), bins=128, cmap="viridis", show_axis=False):
    assert img.ndim == 2
    plt.figure(figsize=size)

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap=cmap)
    plt.title("Original")
    plt.axis("equal")
    if not show_axis:
        plt.axis("off")

    # Contour
    plt.subplot(1, 3, 2)
    plt.contour(img, origin="image", cmap=cmap)
    plt.title("Contour")
    plt.axis("equal")
    if not show_axis:
        plt.axis("off")

    # Histgram
    plt.subplot(1, 3, 3)
    plt.hist(img.flatten(), bins)
    plt.title("Histgram")
    plt.show()


def show_histgrams(img_list, title_list=None, bins=256, cmap="viridis", size=(16, 8)):
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
