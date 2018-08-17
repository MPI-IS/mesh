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

   Mesh<mesh>
   Mesh Viewer<mesh_viewer>
   Geometry<geometry>


What is this package about?
===========================
This package contains core functions for manipulating Meshes and visualizing them.


Getting started
===============

Installation
------------

1. Get the last copy of the ``psbody-mesh`` package (ask around you, or go to our wiki to have an indication on
   where to start)

  a. for Linux, take a source distribution
  b. for OSX, take a wheel distribution (ending with ``.whl``)

2. If you are on Linux, the source distribution needs to compile the package. For the compilation to succeed,
   `Boost`_ is currently required, but only its **header** part. You may install Boost like this::

     sudo apt-get install libboost-dev

   or just download the last archive from `Boost`_ (version >= 1.57 required).

3. Then finally just type the following::

     virtualenv my_venv
     . my_venv/bin/activate
     pip install -U pip
     pip install wheel_or_source_package

   For the source package, if Boost is not installed system wise, but deflated from an archive taken from `Boost`_,
   you may specify the location of the headers like this::

     pip install --install-option="--boost-location=location/of/boost/include/folder" psbody_mesh_XXX.tar.gz

   .. note::

     If ``pip`` tries to install the dependencies such as ``numpy`` while those are already installed in your (virtual)
     environment, you can just add the ``--no-deps`` option to the previous command line to prevent that.

   .. note::

     Please note that using ``pip`` directly will work "better" that typing ``python setup.py install`` in the
     sense that ``pip`` will install all the dependencies in one command.

4. If you are conscientious, you may also run the unit tests to check that the installation did go well. The unittest
   files are inside the source archive, so you need to download the archive and deflate it somewhere to run the tests
   after having installed the Mesh package.
   Then running the tests is just a matter of typing the following commands::

     tar xzf psbody_mesh_XXX.tar.gz
     cd psbody_mesh_XXX
     python -m unittest discover -vvv .

   .. note::

     The unit tests use external files that are currently on a Max Planck Institute shared folder. The location of the
     folder is hard-coded in the file ``tests/__init__.py``, you may need to adapt this location with your own setup, or set
     the environment variable ``PSBODY_TEST_DATA_FOLDER`` to the right location (look at the ``tests/__init__.py`` for
     more information).

.. _Boost: http://www.boost.org

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

Running inside a virtual environment
------------------------------------
*Beware*: there is no problem in running the Mesh package inside a virtual environment. However,
if you want to run within an interactive shell such as **IPython**, you **have** to install IPython
inside the virtual environment as well::

	In [1]: from psbody.mesh import MeshViewers

	In [2]: mviewer = MeshViewers(shape=(2,2))
	OpenGL test failed:
	        stdout:
	        stderr: /usr/bin/python: No module named psbody.mesh


As shown above, the python binary is not the correct one. If after having installed IPython inside the
virtual environment, you still encounter this issue, then it is likely that your bash is caching the
``ipython`` command to the wrong one. Example::

    >> type ipython
    ipython is hashed (/usr/bin/ipython)
    >> which ipython
    /media/renficiaud/linux-data/Code/sandbox/venv_meshviewer/bin/ipython
    >> hash -r # clears the cache
    >> type ipython
    /media/renficiaud/linux-data/Code/sandbox/venv_meshviewer/bin/ipython


Caching
=======

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
