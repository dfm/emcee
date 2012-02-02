VERSION = 1.0.0
OUT_DIR = docs/${VERSION}

docs:
	mkdir -p ${OUT_DIR}
	pycco -d ${OUT_DIR} emcee/*.py

.PHONY: docs

