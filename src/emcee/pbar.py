# -*- coding: utf-8 -*-

import logging

__all__ = ["get_progress_bar"]

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
    )
except ImportError:
    Progress = None


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


class _RichPBar(object):
    """A wrapper that provides emcee's progress-bar interface over rich."""

    def __init__(self, total, **kwargs):
        self.total = total
        self.description = kwargs.pop("desc", "Sampling")
        leave = kwargs.pop("leave", True)
        self.progress = None
        self.task_id = None

        # leave=False means clearing the bar when complete.
        self.transient = not leave

        # Preserve legacy behavior by writing to stderr by default.
        self.console = kwargs.pop("console", Console(stderr=True))

        if kwargs:
            logger.warning(
                "Ignoring unsupported progress bar kwargs for rich backend: %s",
                ", ".join(sorted(kwargs.keys())),
            )

    def __enter__(self, *args, **kwargs):
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=self.transient,
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task(
            self.description, total=self.total
        )
        return self

    def __exit__(self, *args, **kwargs):
        self.progress.__exit__(*args, **kwargs)

    def update(self, count):
        self.progress.update(self.task_id, advance=count)


def get_progress_bar(display, total, **kwargs):
    """Get a progress bar interface with given properties

    If the rich library is not installed, this will always return a "progress
    bar" that does nothing.

    Args:
        display (bool or str): Should the bar actually show the progress?
        total (int): The total size of the progress bar.
        kwargs (dict): Optional keyword arguments to be passed to the progress
            bar implementation.

    """
    if display:
        if Progress is None:
            logger.warning(
                "You must install the rich library to use progress "
                "indicators with emcee"
            )
            return _NoOpPBar()
        else:
            return _RichPBar(total=total, **kwargs)

    return _NoOpPBar()
