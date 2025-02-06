# Makefile for mesh package
package_name := mesh_package

all:
	@echo "\033[0;36m----- [" ${package_name} "] Installing with the interpreter `which python` (version `python --version | cut -d' ' -f2`)\033[0m"
	@pip install --upgrade -r requirements.txt && pip list
	@BOOST_ROOT=${BOOST_INCLUDE_DIRS} pip install --no-deps --verbose --no-cache-dir .

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

sdist:
	@echo "\033[0;33m----- [" ${package_name} "] Creating the source distribution\033[0m"
	@python setup.py sdist

wheel:
	@echo "\033[0;33m----- [" ${package_name} "] Creating the wheel distribution\033[0m"
	@pip install wheel
	@BOOST_ROOT=${BOOST_INCLUDE_DIRS} python setup.py --verbose build_ext bdist_wheel

documentation:
	@echo "\033[0;33m----- [" ${package_name} "] Building Sphinx documentation\033[0m"
	@pip install -U sphinx sphinx_bootstrap_theme
	@cd doc && make html

clean:
	@rm -rf build
	@rm -rf dist
	@rm -rf psbody_mesh.egg-info
	@rm -rf *.xml
