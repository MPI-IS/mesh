# (c) 2015-2016 Max Planck Society
# see accompanying LICENSE.txt file for licensing and contact information

try:
    # setuptools is required
    from setuptools import setup, Extension as _Extension
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.install import install as _install

    has_setup_tools = True
except ImportError:
    from distutils.core import setup, Extension as _Extension
    from distutils.command.build_ext import install as _install

    has_setup_tools = False

from distutils.util import convert_path
from distutils.core import Command
from distutils import log
from distutils.command.sdist import sdist as _sdist

import os

# this package will go to the following namespace
namespace_package = 'psbody'

# the CGAL archive
CGAL_archive = convert_path('mesh/thirdparty/CGAL-4.7.tar.gz')


def _get_version():
    """Convenient function returning the version of this package"""

    ns = {}
    version_path = convert_path('mesh/version.py')
    if not os.path.exists(version_path):
        return None
    with open(version_path) as version_file:
        exec(version_file.read(), ns)

    log.warn('[VERSION] read version is %s', ns['__version__'])
    return ns['__version__']


class build_deflate_cgal(Command):
    """Deflates CGal to a temporary build folder"""

    description = "deflate CGAL"
    # option with '=' because it takes an argument
    user_options = [('cgal-location=', None, 'specifies the location of the cgal archive (tar.gz file)'),
                    ]

    def initialize_options(self):
        self.build_temp = None

    def finalize_options(self):
        self.set_undefined_options('build', ('build_temp', 'build_temp'),)
        pass

    def run(self):

        CGAL_dir_deflate = os.path.abspath(self.build_temp)

        log.info('[CGAL] deflating cgal from "%s" to "%s"', CGAL_archive, CGAL_dir_deflate)
        if not os.path.exists(os.path.join(CGAL_dir_deflate, 'CGAL-4.7')):
            import tarfile
            os.makedirs(CGAL_dir_deflate)

            cgal_tar = tarfile.open(CGAL_archive, 'r:*')
            cgal_tar.extractall(CGAL_dir_deflate)

        # create a dummy configuration file
        config_file = os.path.join(CGAL_dir_deflate, 'CGAL-4.7', 'include', 'CGAL', 'compiler_config.h')
        if not os.path.exists(config_file):
            open(config_file, 'w')

        pass


class build_ext(_build_ext):
    """We override the regular extension processing to add our own dependencies"""

    user_options = [('boost-location=', None, 'specifies the location of the boost folder (only include needed)'),
                    ] + _build_ext.user_options

    def initialize_options(self):
        self.boost_location = None
        return _build_ext.initialize_options(self)

    def finalize_options(self):

        self.set_undefined_options('install', ('boost_location', 'boost_location'),)
        if self.boost_location is not None and self.boost_location.strip():
            # avoid empty folder name as it may happen and mess with the compiler
            #
            # we cannot assert that boost_location exist here, because we are
            # running this code for targets that do not require compilation
            # such as sdist

            # check for subfolders in the boost-x-yy-z sense
            # check for env variables
            self.boost_location = os.path.expanduser(self.boost_location)

        return _build_ext.finalize_options(self)

    def build_extension(self, ext):
        """Adds the necessary include folders"""

        # should be possible to have boost on the system
        # assert(self.boost_location is not None), 'the boost location should be provided with the option "--boost-location"'

        ext.include_dirs += [os.path.join(os.path.abspath(self.build_temp), 'CGAL-4.7', 'include')]
        if self.boost_location is not None:
            ext.include_dirs += [self.boost_location]

        # Remove empty paths
        filtered = []
        for in_dir in filter(None, ext.include_dirs):
            filtered.append(in_dir)
        ext.include_dirs = filtered

        return _build_ext.build_extension(self, ext)

    def run(self):
        """Runs the dependant targets"""
        # the 1 at the end construct the object always, even if not specified on
        # the command line.
        build_deflate_cgal = self.get_finalized_command('build_deflate_cgal', 1)
        build_deflate_cgal.run()

        return _build_ext.run(self)

    # see subcommands documentation in the original Command class
    sub_commands = [('build_deflate_cgal', None)] + _build_ext.sub_commands


class install(_install):
    """We override the regular extension processing to add our own dependencies"""

    user_options = [('boost-location=', None, 'specifies the location of the boost folder (only include needed)'),
                    ] + _install.user_options

    def initialize_options(self):
        self.boost_location = None
        return _install.initialize_options(self)

    def finalize_options(self):

        # if self.boost_location is not None:
        #     self.boost_location = os.path.expanduser(self.boost_location)

        return _install.finalize_options(self)


class sdist(_sdist):
    """Modified source distribution that adds the CGAL distribution to the generated package"""

    def get_file_list(self):
        """Extends the file list read from the manifest with the sources of Yayi"""

        _sdist.get_file_list(self)

        # including the CGal archive without being forced to use the Manifest
        self.filelist.append(CGAL_archive)

        # distributing the tests files without being forced to use the Manifest
        for i in os.listdir(convert_path('tests')):
            if os.path.splitext(i)[1] == ".py":
                self.filelist.append(convert_path(os.path.join('tests', i)))

        log.info('[SDIST] file list is:')
        for f in self.filelist.files:
            log.info('[SDIST] \t"%s"', f)

        return


def _get_all_extensions():
    try:
        import numpy
    except:
        return []

    # valid only for gcc/clang
    extra_args = ['-O3']

    import sys
    if sys.platform.find('linux') > -1:
        extra_args += ['-fopenmp']  # openmp not supported on OSX

    define_macros = [('NDEBUG', '1')]

    define_macros_mesh_ext_without_cgal_link = [
        ('CGAL_NDEBUG', 1),
        ('MESH_CGAL_AVOID_COMPILED_VERSION', 1),
        ('CGAL_HAS_NO_THREADS', 1),
        ('CGAL_NO_AUTOLINK_CGAL', 1)
    ]

    undef_macros = []

    package_name_and_srcs = [('aabb_normals', ['mesh/src/aabb_normals.cpp'], define_macros_mesh_ext_without_cgal_link),
                             ('spatialsearch', ['mesh/src/spatialsearchmodule.cpp'], define_macros_mesh_ext_without_cgal_link),
                             ('visibility', ['mesh/src/py_visibility.cpp', 'mesh/src/visibility.cpp'], define_macros_mesh_ext_without_cgal_link),
                             ('serialization.plyutils', ['mesh/src/plyutils.c', 'mesh/src/rply.c'], []),
                             ('serialization.loadobj', ['mesh/src/py_loadobj.cpp'], []),
                             ]

    out = []

    for current_package_name, src_list, additional_defines in package_name_and_srcs:
        ext = _Extension("%s.mesh.%s" % (namespace_package, current_package_name),
                         src_list,
                         language="c++",
                         include_dirs=['mesh/src', numpy.get_include()],
                         libraries=[],
                         define_macros=define_macros + additional_defines,
                         undef_macros=undef_macros,
                         extra_compile_args=extra_args,
                         extra_link_args=extra_args)

        out += [ext]

    return out

all_extensions = _get_all_extensions()

additional_kwargs = {}
if has_setup_tools:
    # setup tools required for the 'setup_requires' ...
    additional_kwargs['setup_requires'] = ['setuptools', 'numpy']
    additional_kwargs['install_requires'] = [
        'numpy >= 1.8',
        'opencv-python',
        'pillow',
        'pyopengl',
        'pyyaml',
        'pyzmq',
        'scipy',
    ]
    additional_kwargs['zip_safe'] = not all_extensions
    additional_kwargs['test_suite'] = "tests"
    additional_kwargs['namespace_packages'] = [namespace_package]

cmdclass = {'build_ext': build_ext,
            'build_deflate_cgal': build_deflate_cgal,
            'sdist': sdist,
            'install': install}

# check if the namespace  works for python >= 3.3
packages = [namespace_package,
            '%s.mesh' % namespace_package,
            '%s.mesh.topology' % namespace_package,
            '%s.mesh.geometry' % namespace_package,
            '%s.mesh.serialization' % namespace_package
            ]  # actual subpackage described here

package_dir = {namespace_package: '%s-mesh-namespace' % namespace_package,
               '%s.mesh' % namespace_package: 'mesh',  # actual subpackage described here
               '%s.mesh.topology' % namespace_package: 'mesh/topology',
               '%s.mesh.geometry' % namespace_package: 'mesh/geometry',
               '%s.mesh.serialization' % namespace_package: 'mesh/serialization',
               }

setup(name='%s-mesh' % namespace_package,
      version=_get_version(),
      packages=packages,
      package_dir=package_dir,
      ext_modules=all_extensions,
      author='Max Planck Perceiving Systems - Body Group',
      maintainer='Jean-Claude Passy',
      maintainer_email='jean-claude.passy@tuebingen.mpg.de',
      url='http://ps.is.tuebingen.mpg.de',
      description='Mesh and MeshViewer utilities',
      license='See LICENSE.txt',
      cmdclass=cmdclass,
      scripts=[
          "bin/meshviewer"
      ],
      ** additional_kwargs
      )
