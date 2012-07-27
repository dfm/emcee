dfm Sphinx Style
================

This repository contains sphinx styles based on Kenneth Reitz's modifications
to  Mitsuhiko's Flask theme.


Usage
-----

1. Put this folder as `_themes` into your docs folder.

2. Add this to your ``conf.py``: ::

	sys.path.append(os.path.abspath('_themes'))
	html_theme_path = ['_themes']
	html_theme = 'dfm'
