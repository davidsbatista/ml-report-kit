test:
	python -m pytest -v tests/

clean:
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .pytest_cache src/*.egg-info

publish_tmp:
	uv pip install -U build	# upgrade build
	python -m build
	uv pip install -U twine # upgrade twine
	python -m twine upload --verbose --repository testpypi dist/*


publish_prod:
	uv pip install -U build # upgrade build
	python -m build
	uv pip install -U twine # upgrade twine
	python -m twine upload --verbose --repository pypi dist/*
