from pathlib import Path


def get_image_pathlist(dirpath):
    jpg_list = list(Path(dirpath).glob("*.jpg"))
    png_list = list(Path(dirpath).glob("*.png"))
    jpg_list.extend(png_list)
    return jpg_list
