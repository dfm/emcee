# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["get_progress_bar"]

import logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class null_pbar(object):
    def __init__(self, total=None):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


def get_progress_bar(display, total):
    if not display:
        return null_pbar()
    if tqdm is None:
        logging.warn("You must install the tqdm library to use progress "
                     "indicators with emcee")
        return null_pbar()
    return tqdm(total=total)
