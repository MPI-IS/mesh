.. PSBody Mesh Package documentation master file, created by
   sphinx-quickstart on Wed Mar 23 07:26:06 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PSBody Mesh Package's documentation!
===============================================

This is the documentation of the Mesh package of the Body Group, Max Planck Institute
for Intelligent Systems, Tübingen, Germany.

Contents:

.. toctree::
   :maxdepth: 2

   Mesh<pages/mesh>
   Mesh Viewer<pages/mesh_viewer>
   Geometry<pages/geometry>


What is this package about?
===========================
This package contains core functions for manipulating Meshes and visualizing them.
It requires ``Python 3.5+`` and is supported on Linux and macOS operating systems.


Getting started
===============

Installation
------------


There are several places where you can download the latest release of the ``psbody-mesh`` package:

* `Code Doc <https://code.is.localnet/series/3/8/>`_  , the internal documentation center of the MPI-IS
* `GitLab <https://gitlab.tuebingen.mpg.de/ps-body/mesh>`_, the internal repository used for development
* `GitHub <https://github.com/MPI-IS/mesh>`_ for the public release

``Code Doc`` contains the wheel and source distributions, and the documentation of the **complete** package.

``GitLab`` contains the source code of the **complete** package.

``GitHub`` contains the source code of the public, **limited** package.


First, create a dedicated Python virtual environment and activate it:

.. code::

    $ python3 -m venv --copies my_venv
    $ source my_venv/bin/activate

The easiest way to install the ``psbody-mesh`` package is to use the wheel distribution:

.. code::

    $ pip install psbody_mesh_*.whl

.. warning::

	Make sure to use to wheel corresponding to your OS and your Python version.

You can also install the ``psbody-mesh`` package using the source distribution.
For this, you first need to install the `Boost <http://www.boost.org>`_ libraries.
You can compile your own local version or simply do:

.. code::

	$ sudo apt-get install libboost-dev

and then install the  ``psbody-mesh`` package:

.. code::

	$ pip install psbody_mesh_*.tar.gz

As a last option, you can also compile and install the ``psbody-mesh`` package using the Makefile.
If you are using the system-wide ``Boost libraries``:

.. code::

	$ make all

or the libraries locally installed:

.. code::

	$ BOOST_ROOT=/path/to/boost/libraries make all

Testing
-------

To run the tests (only available in the **complete** package), simply do:

.. code::

	$ make tests

Documentation
-------------

A detailed documentation can be compiled using the Makefile:

.. code::

	$ make documentation

Loading a mesh
--------------

Loading a :py:class:`Mesh <psbody.mesh.mesh.Mesh>` class from a file is that easy:

.. code::

    from psbody.mesh import Mesh
    my_mesh = Mesh(filename='mesh_filename.ply')

Rendering a mesh
----------------

From a previously loaded mesh ``my_mesh``, it is possible to visualize it inside an interactive window using the
:py:class:`MeshViewers <psbody.mesh.meshviewer.MeshViewers>` class:

.. code::

    from psbody.mesh import MeshViewers

    # creates a grid of 2x2 mesh viewers
    mvs = MeshViewers(shape=[2, 2])

    # sets the first (top-left) mesh to my_mesh
    mvs[0][0].set_static_meshes([my_mesh])

Caching
-------

Some operations make use of caching for performance reasons. The default folder used for caching is

.. code::

  ~/.psbody/mesh_package_cache


If you need to specify the cache folder, define the environment variable ``PSBODY_MESH_CACHE``
prior to any loading of the Mesh package:

.. code::

  export PSBODY_MESH_CACHE="some/folder"
  python
  >> from psbody.mesh import Mesh
  # now uses the specified cache



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
