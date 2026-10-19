import pytest

from emcee.pbar import (
    _ElapsedTimeColumn,
    _NoOpPBar,
    _RemainingTimeColumn,
    _RichPBar,
    get_progress_bar,
)

try:
    import rich
except ImportError:
    rich = None


def test_display_false():
    assert isinstance(get_progress_bar(False, 100), _NoOpPBar)


@pytest.mark.skipif(rich is None, reason="rich not available")
def test_rich_modes():
    assert isinstance(get_progress_bar(True, 1000), _RichPBar)
    assert isinstance(get_progress_bar("std", 1000), _RichPBar)
    assert isinstance(get_progress_bar("notebook", 1000), _RichPBar)
    assert isinstance(get_progress_bar("auto", 1000), _RichPBar)
    assert isinstance(get_progress_bar("autonotebook", 1000), _RichPBar)


@pytest.mark.skipif(rich is None, reason="rich not available")
def test_rich_progress_includes_elapsed_and_remaining():
    pbar = get_progress_bar(True, 1000)

    with pbar:
        assert any(
            isinstance(column, _ElapsedTimeColumn)
            for column in pbar.progress.columns
        )
        assert any(
            isinstance(column, _RemainingTimeColumn)
            for column in pbar.progress.columns
        )
