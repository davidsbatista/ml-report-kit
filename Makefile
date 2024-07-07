test:
	python -m pytest -v tests/

clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache src/*.egg-info

publish_tmp:
	python -m pip install --upgrade build
	python -m build
	python -m pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*


publish_prod:
	python -m pip install --upgrade build
	python -m build
	python -m pip install --upgrade twine
	python -m twine upload --repository pypi dist/*