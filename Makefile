# Development workflow is based on:
# https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html

PATH_DIST=dist


.PHONY: develop
develop:
	pip install build twine
	pip install -e .


.PHONY: build
build:
	rm -r $(PATH_DIST)
	python -m build -n


.PHONY: check
check:
	twine check $(PATH_DIST)/*


.PHONY: publish
publish:
	twine upload --verbose $(PATH_DIST)/*


.PHONY: publish-test
publish-test:
	twine upload --repository testpypi --verbose $(PATH_DIST)/*


.PHONY: install
install:
	pip install .
