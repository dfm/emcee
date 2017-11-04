# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["get_progress_bar"]

import logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class _NoOpPBar(object):
    """This class implements the progress bar interface but does nothing"""
    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


def get_progress_bar(display, total):
    """Get a progress bar interface with given properties

    If the tqdm library is not installed, this will always return a "progress
    bar" that does nothing.

    Args:
        display (bool): Should the bar actually show the progress?
        total (int): The total size of the progress bar.

    """
    if not display:
        return _NoOpPBar()
    if tqdm is None:
        logging.warn("You must install the tqdm library to use progress "
                     "indicators with emcee")
        return _NoOpPBar()
    return tqdm(total=total)
