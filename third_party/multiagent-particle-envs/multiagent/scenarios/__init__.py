import os.path as osp
from importlib.machinery import SourceFileLoader


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return SourceFileLoader("", pathname).load_module()
