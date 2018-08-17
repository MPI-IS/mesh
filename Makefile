# Copyright (c) 2018 Max Planck Society for non-commercial scientific research
# This file is part of psbody.mesh project which is released under MPI License.
# See file LICENSE.txt for full license details.

tmpdirbuild := temporary_test
venv_dir := $(tmpdirbuild)/venv
activate := $(venv_dir)/bin/activate
package_name := mesh_package

.DEFAULT_GOAL := all

# we have to add an option for locating boost on some platforms
ifeq ($(OS),Windows_NT)
  $(error not supported platform)
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Linux)
    libboost_version := $(shell dpkg -l libboost-dev | tail -n 1 | awk '{print $$3}' )
    check_boost_version := $(shell dpkg --compare-versions "${libboost_version}" ">=" 1.54 ; echo $$?)
    ifneq ($(check_boost_version), 0)
      additional_install_cmd := --boost-location=$$BOOST_ROOT
    else
      additional_install_cmd :=
    endif
  else ifeq ($(UNAME_S),Darwin)
    additional_install_cmd := --boost-location=$$BOOST_ROOT
  else
    $(error "NOT SUPPORTED")
  endif
endif

ifneq ($(additional_install_cmd),)
  $(info [INSTALL] Overriding the default install command with [${additional_install_cmd}])
  # cannot use install otherwise it installs it on bdist_wheel command and messes with
  # the virtualenv. build_ext and install use the same "boost-location" option
  additional_install_cmd_for_setup := build_ext ${additional_install_cmd}
  additional_install_cmd_for_pip := --install-option="${additional_install_cmd}"
else
  additional_install_cmd_for_setup :=
  additional_install_cmd_for_pip :=
endif
#$(shell exit 0)

$(tmpdirbuild):
	mkdir -p $(tmpdirbuild)

$(tmpdirbuild)/package_creation: $(tmpdirbuild)
	@echo "\033[0;36m"
	@echo "********"
	@echo "********"
	@echo "********" $(package_name) ": Building the virtualenv for installation"
	@echo "********"
	@echo "********"
	@echo "\033[0m"
	@virtualenv --system-site-packages $(venv_dir)
	@ . $(activate) && pip install --upgrade pip
	@ . $(activate) && pip install --upgrade virtualenv
	@ . $(activate) && pip install nose2
	@ . $(activate) && echo `which pip` && pip -V
	@ . $(activate) && pip install --upgrade setuptools && echo `which pip` && pip -V
	@ . $(activate) && pip install --upgrade wheel && echo `which pip` && pip -V
	@ . $(activate) && pip install numpy scipy pyopengl pillow pyzmq pyyaml && echo `which pip` && pip -V
	####### PACKAGE: creating SDIST target
	@echo "\033[0;33m----- [" ${package_name} "] Creating the source distribution\033[0m"
	@ . $(activate) && python setup.py sdist
	####### PACKAGE: creating WHEEL target
	@echo "\033[0;33m----- [" ${package_name} "] Creating the wheel distribution\033[0m"
	@ . $(activate) && python setup.py --verbose ${additional_install_cmd_for_setup} bdist_wheel
	####### INSTALLING
	@echo "\033[0;33m----- [" ${package_name} "] Installing on the local virtual environment\033[0m"
	@ . $(activate) && pip install --no-deps ${additional_install_cmd_for_pip} --verbose --no-cache-dir .
	####### Cleaning some artifacts
	@rm -rf psbody_mesh.egg-info
	####### Touching the result
	@touch $@

all: $(tmpdirbuild)/package_creation

# TEST_ROOT_DIR is given by the parent process, and points to the location where the xml files should be stored.
# In our case, it also contains the "body" package which is needed by the package

import_tests:
	# not hidding the command of this line
	@echo "********" $(package_name) ": performing import tests"
	unset PYTHONPATH && . $(activate) && PSBODY_MESH_CACHE=`mktemp -d -t mesh_package.XXXXXXXXXX` python -c "from psbody.mesh.mesh import Mesh" && python -c "from psbody.mesh.meshviewer import MeshViewers" ;
	@echo "********" $(package_name) ": performing import tests - ok"

# this package is independant from the rest of the repository, we do not set the PYTHONPATH to
# any other directory.
unit_tests:
	@if test "$(USE_NOSE)" = "" ; then \
		echo "********" $(package_name) ": Running tests using unittest, no report file" ; \
		. $(activate) && PSBODY_MESH_CACHE=`mktemp -d -t mesh_package.XXXXXXXXXX` python -m unittest discover -vvv ; \
	else \
		echo "********" $(package_name) ": Running tests using Nose2, report file is $(TEST_ROOT_DIR)nose2_$(package_name).xml" ; \
		. $(activate) && PSBODY_MESH_CACHE=`mktemp -d -t mesh_package.XXXXXXXXXX` nose2 -v ; \
		echo "********" $(package_name) ": Running tests using Nose2 - ok" ; \
	fi ;


# tests the packages installation
packages_tests:
	@echo "********" $(package_name) ": Building the virtualenv for checking source distribution"
	@rm -rf $(tmpdirbuild)/venv_sdist ;
	@virtualenv $(tmpdirbuild)/venv_sdist ;
	@ . $(tmpdirbuild)/venv_sdist/bin/activate && pip install --upgrade pip ;
	cd dist && dirsdist=$$(python -c "import os; print [i[:os.path.basename(i).find('.tar')]  for i in os.listdir('.') if os.path.basename(i).find('.tar')>-1][0] ") && cd - ; \
	# we do not want to install those packages with potential "additional_install_cmd_for_pip" options ; \
	. $(tmpdirbuild)/venv_sdist/bin/activate && pip install numpy pyopengl pyzmq scipy pyyaml ; \
	. $(tmpdirbuild)/venv_sdist/bin/activate && cd dist && tar xzf *.tar.gz && cd $$dirsdist && pip install ${additional_install_cmd_for_pip} --verbose --no-cache-dir . ;
	@
	@
	@echo "********" $(package_name) ": Building the virtualenv for checking wheel distribution"
	@
	@rm -rf $(tmpdirbuild)/venv_wheel ;
	@virtualenv $(tmpdirbuild)/venv_wheel ;
	@ . $(tmpdirbuild)/venv_wheel/bin/activate && pip install --upgrade pip ;
	@ . $(tmpdirbuild)/venv_wheel/bin/activate && cd dist && pip install --verbose --no-cache-dir *.whl ;

# TODO add a simple and robust way to run the import_tests target
test: import_tests unit_tests packages_tests


install:
	@echo "********" $(package_name) ": installation"
	@echo "** installing " $(package_name) && cd dist && pip install --verbose --no-cache-dir *.whl ;

clean:
	@rm -rf $(tmpdirbuild)
	@find . -name "*.pyc" -delete
	@rm -rf build
	@rm -rf dist
	@rm -rf psbody_mesh.egg-info
