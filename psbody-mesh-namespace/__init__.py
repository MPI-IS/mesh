# this is the setup tools way
__import__('pkg_resources').declare_namespace(__name__)

# this is the distutils way, but does not work with setuptools
#from pkgutil import extend_path
#__path__ = extend_path(__path__, __name__)
