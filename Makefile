docs:
	cd docs;make dirhtml
	git checkout gh-pages
	cp -r docs/_build/dirhtml/* current
	git add .
	git commit -am "Updating docs"
	git push origin gh-pages
	git checkout master
	rm -rf current

.PHONY: docs
