# Perceiving Systems Mesh Package
Mesh processing libraries that support several of our projects such as
* [CoMA: Convolutional Mesh Encoders for Generating 3D Faces](http://coma.is.tue.mpg.de/)
* [FLAME: Learning a model of facial shape and expression from 4D scans](http://flame.is.tue.mpg.de/)
* [MANO: Modeling and Capturing Hands and Bodies Together](http://mano.is.tue.mpg.de/)
* [SMPL: A Skinned Multi-Person Linear Model](http://smpl.is.tue.mpg.de/)
* [VOCA: Voice Operated Character Animation](https://github.com/TimoBolkart/voca)
* [RingNet: 3D Face Shape and Expression Reconstruction from an Image](https://github.com/soubhiksanyal/RingNet)

## Prerequisites
Install `boost`.

On Linux:
```bash
apt-get install libboost-all-dev
```
On Mac OS:
```bash
brew install boost
```



## Installation
It is recommended to install the mesh package in a [virtual environment](https://virtualenv.pypa.io/en/stable/). This package has been tested with Python2.7 on Linux and OSX.
```bash
make
make install
```

## License
Please refer for LICENSE.txt for using this software. The software is compiled using CGAL sources following the license in CGAL_LICENSE.pdf
