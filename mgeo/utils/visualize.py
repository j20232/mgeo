import matplotlib.pyplot as plt


def show_thumbnail(np_img, size=(6, 6)):
    plt.figure(figsize=size)
    plt.axis("off")
    plt.imshow(np_img)
    plt.show()
