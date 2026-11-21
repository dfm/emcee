import pytest

from emcee.pbar import _NoOpPBar, get_progress_bar

try:
    import tqdm
except ImportError:
    tqdm = None


def test_display_false(capsys, caplog):
    pbar = get_progress_bar(False, 100)
    assert isinstance(pbar, _NoOpPBar)

    with pbar as entered:
        assert entered is pbar
        assert entered.update(1) is None
        assert entered.set_description("sampling", False) is None
        assert entered.set_description(desc="sampling", refresh=False) is None

    assert capsys.readouterr() == ("", "")
    assert not caplog.records


@pytest.mark.skipif(tqdm is None, reason="tqdm not available")
def test_tqdm_modes():
    assert isinstance(get_progress_bar(True, 1000), tqdm.asyncio.tqdm_asyncio)
    assert isinstance(get_progress_bar("std", 1000), tqdm.std.tqdm)
    assert isinstance(
        get_progress_bar("notebook", 1000), tqdm.notebook.tqdm_notebook
    )
    assert isinstance(
        get_progress_bar("auto", 1000), tqdm.asyncio.tqdm_asyncio
    )
    assert isinstance(get_progress_bar("autonotebook", 1000), tqdm.std.tqdm)
