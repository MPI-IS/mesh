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

You can download the latest release of the ``psbody-mesh`` package
from the projects `GitHub repository <https://github.com/MPI-IS/mesh>`_.
To install, first you should create a dedicated Python virtual
environment and activate it:

.. code::

    $ python3 -m venv --copies my_venv
    $ source my_venv/bin/activate

To compile the binary extensions you will need to install the `Boost
<http://www.boost.org>`_ libraries.  You can compile your own local
version or simply do:

.. code::

	$ sudo apt-get install libboost-dev

and then compile and install the ``psbody-mesh`` package using the
Makefile.  If you are using the system-wide ``Boost libraries``:

.. code::

	$ make all

or the libraries locally installed:

.. code::

	$ BOOST_INCLUDE_DIRS=/path/to/boost/include make all

Testing
-------

To run the tests simply do:

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

Mesh Viewer
-----------

``meshviewer`` is a program that allows you to display polygonal meshes produced by ``mesh`` package.

Viewing a mesh on a local machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward use-case is viewing the mesh on the same machine where it is stored.  To do this simply run

.. code::

   $ meshviewer view sphere.obj

This will create an interactive window with your mesh rendering. You can render more than one mesh in the same window by passing several paths to ``view`` command

.. code::

   $ meshviewer view sphere.obj cylinder.obj

This will arrange the subplots horizontally in a row.  If you want a grid arrangement, you can specify the grid parameters explicitly

.. code::

   $ meshviewer view -nx 2 -ny 2 *.obj

Viewing a mesh from a remote machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to view a mesh stored on a remote machine.  To do this you need mesh to be installed on both the local and the remote machines.  You start by opening an empty viewer window listening on a network port

.. code::

   (local) $ meshviewer open --port 3000

To stream a shape to this viewer you have to either pick a port that is visible from the remote machine or by manually exposing the port when connecting.  For example, through SSH port forwarding

.. code::

   (local) $ ssh -R 3000:127.0.0.1:3000 user@host

Then on a remote machine you use ``view`` command pointing to the locally forwarded port

.. code::

   (remote) $ meshviewer view -p 3000 sphere.obj

This should display the remote mesh on your local viewer. In case it does not it might be caused by the network connection being closed before the mesh could be sent. To work around this one can try increasing the timeout up to 1 second

.. code::

   (remote) $ meshviewer view -p 3000 --timeout 1 sphere.obj

To take a snapshot you should locally run a `snap` command

.. code::

   (local) $ meshviewer snap -p 3000 sphere.png

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
