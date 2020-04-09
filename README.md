Perceiving Systems Mesh Package
===============================

This package contains core functions for manipulating meshes and visualizing them.
It requires ``Python 3.5+`` and is supported on Linux and macOS operating systems.

The ``Mesh`` processing libraries support several of our projects such as
* [CoMA: Convolutional Mesh Encoders for Generating 3D Faces](http://coma.is.tue.mpg.de/)
* [FLAME: Learning a model of facial shape and expression from 4D scans](http://flame.is.tue.mpg.de/)
* [MANO: Modeling and Capturing Hands and Bodies Together](http://mano.is.tue.mpg.de/)
* [SMPL: A Skinned Multi-Person Linear Model](http://smpl.is.tue.mpg.de/)
* [VOCA: Voice Operated Character Animation](https://github.com/TimoBolkart/voca)
* [RingNet: 3D Face Shape and Expression Reconstruction from an Image](https://github.com/soubhiksanyal/RingNet)

Requirements
------------

You first need to install the `Boost <http://www.boost.org>`_ libraries.
You can compile your own local version. This is a must for Windows/MSCV. On Linux you can simply install it via:

```
$ sudo apt-get install libboost-dev
```

On macOS:

```
$ brew install boost
```

Installation
------------

First, create a dedicated Python virtual environment and activate it:

```
$ python3 -m venv --copies my_venv
$ source my_venv/bin/activate
```

#### Linux/MacOS
You should then compile and install the ``psbody-mesh`` package using the Makefile.
If you are using the system-wide ``Boost`` libraries:

```
$ make all
```

or the libraries locally installed:

```
$ BOOST_ROOT=/path/to/boost/libraries make all
```

#### Windows 

The makefile will not work on Windows with MSVC. Not tested with neither CygWin nor MinGw. Execute the following instead:
```
python setup.py install boost-location=<path_to_your_boost>
```

Testing
-------

To run the tests, simply do:

```
$ make tests
```

Documentation
-------------

A detailed documentation can be compiled using the Makefile:

```
$ make documentation
```

License
-------
Please refer for LICENSE.txt for using this software. The software is compiled using CGAL sources following the license in CGAL_LICENSE.pdf

Acknowledgments
---------------

We thank the external contribution from the following people:
* [Kenneth Chaney](https://github.com/k-chaney)  ([PR #5](https://github.com/MPI-IS/mesh/pull/5))
* [Dávid Komorowicz](https://github.com/Dawars) ([PR #8](https://github.com/MPI-IS/mesh/pull/8))