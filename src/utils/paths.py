import os


def uppath(_path: str, n: int):
    return os.sep.join(_path.split(os.sep)[:-n])