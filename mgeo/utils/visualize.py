import matplotlib.pyplot as plt


def show(np_img, points=None, mark="b*", size=(6, 6), to_gray=False, show_axis=False):
    plt.figure(figsize=size)
    if to_gray:
        plt.gray()
    if not show_axis:
        plt.axis("off")
    if points is not None and points.ndim == 2:
        if points.shape[1] == 2:
            plt.plot(points[:, 0], points[:, 1], mark)
        else:
            plt.plot(points[0], points[1], mark)
    plt.imshow(np_img)
    plt.show()


def show_histgram(np_img, size=(15, 6), bins=128, show_axis=False):
    assert np_img.ndim == 2
    plt.figure(figsize=size)
    plt.gray()

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(np_img)
    plt.title("Original")
    plt.axis("equal")
    if not show_axis:
        plt.axis("off")

    # Contour
    plt.subplot(1, 3, 2)
    plt.contour(np_img, origin="image")
    plt.title("Contour")
    plt.axis("equal")
    if not show_axis:
        plt.axis("off")

    # Histgram
    plt.subplot(1, 3, 3)
    plt.hist(np_img.flatten(), bins)
    plt.title("Histgram")
    plt.show()


def show_imgs(img_list, title_list=None, size=(16, 8), to_gray=False, show_axis=False):
    if title_list is not None:
        assert len(img_list) == len(title_list)
    plt.figure(figsize=size)
    if to_gray:
        plt.gray()
    for idx, img in enumerate(img_list):
        plt.subplot(1, len(img_list), idx + 1)
        plt.imshow(img)
        if title_list is not None:
            plt.title(title_list[idx])
        if not show_axis:
            plt.axis("off")
    plt.show()
