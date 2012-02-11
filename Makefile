VERSION = 1.0.0
OUT_DIR = docs/${VERSION}

EXAMPLES = rosenbrock quickstart

default: docs examples

docs:
	mkdir -p ${OUT_DIR}
	pycco -d ${OUT_DIR} emcee/*.py

examples:
	$(foreach e, ${EXAMPLES}, python docs/gen_example.py examples/${e}.py docs/${e}/index.html;)

.PHONY: docs examples

