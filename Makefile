# Makefile for mesh package
package_name := mesh_package

all:
	@echo "\033[0;36m----- [" ${package_name} "] Installing with the interpreter `which python` (version `python --version | cut -d' ' -f2`)\033[0m"
	@pip install --upgrade -r requirements.txt && pip list
	@pip install --no-deps --config-settings="--boost-location=$$BOOST_INCLUDE_DIRS" --verbose --no-cache-dir .

import_tests:
	@echo "\033[0;33m----- [" ${package_name} "] Performing import tests\033[0m"
	@PSBODY_MESH_CACHE=`mktemp -d -t mesh_package.XXXXXXXXXX` python -c "from psbody.mesh.mesh import Mesh"
	@python -c "from psbody.mesh.meshviewer import MeshViewers"
	@echo "\033[0;33m----- [" ${package_name} "] OK import tests\033[0m"

unit_tests:
	@if test "$(USE_NOSE)" = "" ; then \
		echo "\033[0;33m----- [" ${package_name} "] Running tests using unittest, no report file\033[0m" ; \
		PSBODY_MESH_CACHE=`mktemp -d -t mesh_package.XXXXXXXXXX` python -m unittest -v ; \
	else \
		echo "\033[0;33m----- [" ${package_name} "] Running tests using nosetests\033[0m" ; \
		pip install nose ; \
		PSBODY_MESH_CACHE=`mktemp -d -t mesh_package.XXXXXXXXXX` nosetests -v --with-xunit; \
	fi ;

tests: import_tests unit_tests

# Creating source distribution
sdist:
	@echo "\033[0;33m----- [" ${package_name} "] Creating the source distribution\033[0m"
	@python setup.py sdist

# Creating wheel distribution
wheel:
	@echo "\033[0;33m----- [" ${package_name} "] Creating the wheel distribution\033[0m"
	@pip install wheel
	@python setup.py --verbose build_ext --boost-location=$$BOOST_INCLUDE_DIRS bdist_wheel

# Build documentation
documentation:
	@echo "\033[0;33m----- [" ${package_name} "] Building Sphinx documentation\033[0m"
	@pip install -U sphinx sphinx_bootstrap_theme
	@cd doc && make html

clean:
	@rm -rf build
	@rm -rf dist
	@rm -rf psbody_mesh.egg-info
	@rm -rf *.xml
