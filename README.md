[![Build Status](
    https://raw.githubusercontent.com/MPI-IS-BambooAgent/sw_badges/master/badges/plans/ps-body-mesh/tag.svg?sanitize=true)](
        https://atlas.is.localnet/bamboo/browse/PS-FMP/latest)

Perceiving Systems Mesh Package
===============================

This package contains core functions for manipulating meshes and
visualizing them.  It requires ``Python 3.5+`` and is supported on
Linux and macOS operating systems.

The ``Mesh`` processing libraries support several of our projects such as
* [CoMA: Convolutional Mesh Encoders for Generating 3D Faces](http://coma.is.tue.mpg.de/)
* [FLAME: Learning a model of facial shape and expression from 4D scans](http://flame.is.tue.mpg.de/)
* [MANO: Modeling and Capturing Hands and Bodies Together](http://mano.is.tue.mpg.de/)
* [SMPL: A Skinned Multi-Person Linear Model](http://smpl.is.tue.mpg.de/)
* [VOCA: Voice Operated Character Animation](https://github.com/TimoBolkart/voca)
* [RingNet: 3D Face Shape and Expression Reconstruction from an Image](https://github.com/soubhiksanyal/RingNet)
* [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de/)

Requirements
------------

This package requires the `Boost <http://www.boost.org>` libraries in order to work.

You can either create a dedicated Conda virtual environment and install [Boost from Anaconda](https://anaconda.org/anaconda/boost) (see **Installation with Conda**), or compile your own local version and install it globally on Linux with

```bash
$ sudo apt-get install libboost-dev
# the default BOOST_INCLUDE_DIRS path is /usr/include
```

or on macOS

```bash
$ brew install boost
$ brew --prefix boost   # show BOOST_INCLUDE_DIRS path
```

Installation
------------

First, create a dedicated Python virtual environment and activate it:

```bash
$ python3 -m venv --copies my_venv
$ source my_venv/bin/activate
```

You should then compile and install the ``psbody-mesh`` package easily
using the Makefile:

```bash
$ BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```


Installation with Conda
------------

*Note: This guide has been written for and tested on Linux Ubuntu 18.04; however, given its dependence on Conda, it should be (easily) adaptable to other operative systems.*

1. First, create a dedicated Python 3 virtual environment and activate it; note that  you can replace ``my_venv`` with another string (in all of the following commands) in order to give the virtual environment a custom name:

    ```bash
    $ conda create --name my_venv python=3.8
    $ conda activate my_venv
    ```

2. Install the Boost libraries through an Anaconda package:

    ```bash
    $ conda install -c anaconda boost
    ```

3. Clone into the ``psbody-mesh`` repository:

    ```bash
    $ git clone https://github.com/MPI-IS/mesh
    ```

4. Install the ``psbody-mesh`` package easily with ``pip``:

    ```bash
    $ pip install --upgrade -r mesh/requirements.txt
    $ pip install --no-deps --install-option="--boost-location=$$BOOST_INCLUDE_DIRS" --verbose --no-cache-dir mesh/.
    ```

5. Done! Now you can add ``import psbody.mesh`` to any of your Python 3 scripts and execute them in the virtual environment thus created.

Testing
-------

To run the tests, simply do:

```bash
$ make tests
```

Documentation
-------------

A detailed documentation can be compiled using the Makefile:

```bash
$ make documentation
```

Viewing the Meshes
------------------

Starting from version 0.4 meshviewer ships with `meshviewer` -- a
program that allows you to display polygonal meshes produced by `mesh`
package.

### Viewing a mesh on a local machine

The most straightforward use-case is viewing the mesh on the same
machine where it is stored.  To do this simply run

```bash
$ meshviewer view sphere.obj
```

This will create an interactive window with your mesh rendering.  You
can render more than one mesh in the same window by passing several
paths to `view` command

```bash
$ meshviewer view sphere.obj cylinder.obj
```

This will arrange the subplots horizontally in a row.  If you want a
grid arrangement, you can specify the grid parameters explicitly

```bash
$ meshviewer view -nx 2 -ny 2 *.obj
```

### Viewing a mesh from a remote machine

It is also possible to view a mesh stored on a remote machine.  To do
this you need mesh to be installed on both the local and the remote
machines.  You start by opening an empty viewer window listening on a
network port

```bash
(local) $ meshviewer open --port 3000
```

To stream a shape to this viewer you have to either pick a port that
is visible from the remote machine or by manually exposing the port
when connecting.  For example, through SSH port forwarding

```bash
(local) $ ssh -R 3000:127.0.0.1:3000 user@host
```

Then on a remote machine you use `view` command pointing to the
locally forwarded port

```bash
(remote) $ meshviewer view -p 3000 sphere.obj
```

This should display the remote mesh on your local viewer. In case it
does not it might be caused by the network connection being closed
before the mesh could be sent. To work around this one can try
increasing the timeout up to 1 second

```bash
(remote) $ meshviewer view -p 3000 --timeout 1 sphere.obj
```

To take a snapshot you should locally run a `snap` command

```bash
(local) $ meshviewer snap -p 3000 sphere.png
```

License
-------

Please refer for LICENSE.txt for using this software. The software is
compiled using CGAL sources following the license in CGAL_LICENSE.pdf

Acknowledgments
---------------

We thank the external contribution from the following people:
* [Kenneth Chaney](https://github.com/k-chaney)  ([PR #5](https://github.com/MPI-IS/mesh/pull/5))
* [Dávid Komorowicz](https://github.com/Dawars) ([PR #8](https://github.com/MPI-IS/mesh/pull/8))
