import pytest

from emcee.pbar import _NoOpPBar, get_progress_bar

try:
    import tqdm
except ImportError:
    tqdm = None


def test_display_false():
    assert isinstance(get_progress_bar(False, 100), _NoOpPBar)


@pytest.mark.skipif(tqdm is None, reason="tqdm not available")
def test_tqdm_modes():
    assert isinstance(get_progress_bar(True, 1000), tqdm.std.tqdm)
    assert isinstance(
        get_progress_bar("notebook", 1000), tqdm.notebook.tqdm_notebook
    )
    assert isinstance(
        get_progress_bar("auto", 1000), tqdm.asyncio.tqdm_asyncio
    )
    assert isinstance(get_progress_bar("autonotebook", 1000), tqdm.std.tqdm)
