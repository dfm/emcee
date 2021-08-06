# -*- coding: utf-8 -*-

import logging

__all__ = ["get_progress_bar"]

logger = logging.getLogger(__name__)

try:
    import tqdm
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


def get_progress_bar(display, total, **kwargs):
    """Get a progress bar interface with given properties

    If the tqdm library is not installed, this will always return a "progress
    bar" that does nothing.

    Args:
        display (bool or str): Should the bar actually show the progress? Or a
                               string to indicate which tqdm bar to use.
        total (int): The total size of the progress bar.
        kwargs (dict): Optional keyword arguments to be passed to the tqdm call.

    """
    if display:
        if tqdm is None:
            logger.warning(
                "You must install the tqdm library to use progress "
                "indicators with emcee"
            )
            return _NoOpPBar()
        else:
            if display is True:
                return tqdm.tqdm(total=total, **kwargs)
            else:
                return getattr(tqdm, "tqdm_" + display)(total=total, **kwargs)

    return _NoOpPBar()
