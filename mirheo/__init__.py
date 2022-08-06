#!/usr/bin/env python

import inspect
import functools
import sys
import weakref
import re

from libmirheo import *

__all__ = ["version", "tools"]

# Global variable for the mirheo coordination class
# Used in decorators to access compute task status
# This variable made a weak reference to not prevent
# cleanup of the simulation
__coordinator = None

# Global unit converter. Applied to all values passed to Mirheo constructors.
__unit_converter = None

class PintUnitsConverter:
    """Converts given values to the given unit system."""
    UNIT_SYSTEM_NAME = '__mirheo'

    def __init__(self, ureg, mirL='mirL', mirT='mirT', mirM='mirM'):
        # A pint definitions file can be provided as a list of strings.
        definitions = [
            '@system ' + self.UNIT_SYSTEM_NAME,
            '    ' + mirL,
            '    ' + mirT,
            '    ' + mirM,
            '@end',
        ]
        ureg.load_definitions(definitions)
        self.ureg = ureg

    def __call__(self, value):
        """Strip off all units using the Mirheo unit system as a reference.

        All unrecognized data types will be returned as is.
        """
        # Just in case, we use `is` instead of `isinstance`, to avoid
        # accidentally casting to base classes (e.g. OrderedDict -> dict).
        cls = value.__class__
        ureg = self.ureg
        if cls is ureg.Quantity:
            # The complicated (and quite expensive) procedure of changing an
            # arbitrary quantity to the given unit system... We cannot use
            # .m_as() directly as we would need to know the corresponding unit
            # in Mirheo's unit system.
            old = ureg.default_system
            try:
                ureg.default_system = self.UNIT_SYSTEM_NAME
                return value.to_base_units().magnitude
            finally:
                ureg.default_system = old
        if cls is tuple:
            return tuple(self(x) for x in value)
        if cls is list:
            return [self(x) for x in value]
        if cls is dict:
            return {k: self(v) for k, v in value.items()}
        return value


def set_unit_registry(ureg, mirL='mirL', mirT='mirT', mirM='mirM'):
    """Register a pint UnitRegistry and Mirheo's coordinate system.

    The unit registry will be used to convert any values with units to the
    given Mirheo unit system before passing them to the Mirheo C++ functions.

    Arguments:
        ureg: a ``pint.UnitRegistry`` object
        mirL: name of the Mirheo length unit, defaults to ``mirL``
        mirT: name of the Mirheo time unit, defaults to ``mirT``
        mirM: name of the Mirheo mass unit, defaults to ``mirM``
    """
    global __unit_converter
    __unit_converter = PintUnitsConverter(ureg, mirL, mirT, mirM)


def unit_conversion_decorator(o):
    """Decorate a method or all methods of a class with automatic unit conversion."""

    # If o is a class, decorate all methods.
    if isinstance(o, type):
        for key, value in o.__dict__.items():
            if callable(value):
                setattr(o, key, unit_conversion_decorator(value))
        return o

    # Otherwise, assume it is a function.
    @functools.wraps(o)
    def wrapper(*args, **kwargs):
        if __unit_converter:
            args = __unit_converter(args)
            kwargs = __unit_converter(kwargs)

        return o(*args, **kwargs)

    return wrapper


def decorate_object(f, needState = True):
    """
    Wrap the __init__ or __new__ method of all the simulation handlers and
    particle vectors. If we are not a compute task, just return None pass the
    state if needState is True.
    """
    @unit_conversion_decorator
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if __coordinator is None:
            raise Exception('No coordinator created yet!')

        if __coordinator().isComputeTask():
            if needState:
                return f(self, __coordinator().getState(), *args, **kwargs)
            else:
                return f(self, *args, **kwargs)
        else:
            return None
    return wrapper


def decorate_coordinator(f):
    """Wrap the creation of the coordinator."""
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        global __coordinator
        f(self, *args, **kwargs)

        if __coordinator is not None and  __coordinator() is not None:
           raise Exception('There can only be one coordinator at a time!')
       
        __coordinator = weakref.ref(self)

    return wrapper


def decorate_func_with_plugin_arg(f):
    """Decorate a function that takes a plugin as an argument.

    A "plugin" is a pair of simulation and postprocess plugins.
    The decorator expands this pair.
    """
    @functools.wraps(f)
    def wrapper(self, plugins_tuple):
        return f(self, plugins_tuple[0], plugins_tuple[1])

    return wrapper


# Wrap the creation of plugins
# Pass the compute task status into the creation function
# Pass the common global state associated to the coordinator
def decorate_plugins(f):
    @unit_conversion_decorator
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global __coordinator
        if __coordinator is None:
            raise Exception('No coordinator created yet!')

        return f(__coordinator().isComputeTask(),
                 __coordinator().getState(),
                 *args, **kwargs)

    return wrapper


# Make MPI abort the program if an exception occurs
# https://groups.google.com/forum/#!topic/mpi4py/RovYzJ8qkbc
def make_excepthook(old_excepthook):
    def excepthook(exc_type, exc_value, exc_traceback):
        old_excepthook(exc_type, exc_value, exc_traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        if __coordinator is not None and  __coordinator() is not None:
            abort()

    return excepthook


def __init__():
    # Setup exception handling
    sys.excepthook = make_excepthook(sys.excepthook)

    # Wrap everything except for plugins and non-GPU stuff
    # Make the __init__ functions return None if we are not a compute task
    nonGPU_names  = [['Interactions', 'MembraneParameters'],
                     ['Interactions', 'KantorBendingParameters'],
                     ['Interactions', 'JuelicherBendingParameters']]
    
    needing_state = ['Plugins', 'Integrators', 'ParticleVectors',
                     'Interactions', 'BelongingCheckers', 'Bouncers', 'Walls']

    not_needing_state = [['ParticleVectors', 'MembraneMesh'],
                         ['ParticleVectors', 'Mesh']]
    
    classes = {}
    submodules = inspect.getmembers(sys.modules[__name__],
                                    lambda member: inspect.ismodule(member)
                                    and 'mirheo' in member.__name__ )
    for m in submodules:
        classes[m[0]] = inspect.getmembers(sys.modules[m[1].__name__],
                                        lambda member: inspect.isclass(member)
                                        and 'mirheo' in member.__module__ )

    for module in classes.keys():
        if module != 'Plugins':            
            for cls in classes[module]:                
                if [module, cls[0]] not in nonGPU_names:
                    need_state = module in needing_state
                    if [module, cls[0]] in not_needing_state:
                        need_state = False
                    setattr(cls[1], '__init__', decorate_object(cls[1].__init__, need_state))
                    setattr(cls[1], '__new__',  decorate_object(cls[1].__new__ , need_state))
                    getattr(cls[1], '__init__').__doc__ = re.sub('state: libmirheo.MirState, ',
                                                                 '',
                                                                 getattr(cls[1], '__init__')
                                                                 .__doc__)
                    
    # Now wrap plugins creation
    # Also change the names of the function
    # by removing the double underscore
    for m in submodules:
        if m[0] == 'Plugins':
            funcs = inspect.getmembers(sys.modules[m[1].__name__],
                                        lambda member: inspect.isbuiltin(member)
                                        and 'mirheo' in member.__module__)
            
            
            for f in funcs:
                if '__create' in f[0]:
                    newname = f[0][2:]
                    setattr(m[1], newname, decorate_plugins(f[1]))
                    getattr(m[1], newname).__doc__ = re.sub('__' + newname, newname,    getattr(m[1], newname).__doc__)
                    getattr(m[1], newname).__doc__ = re.sub('compute_task: bool, ', '', getattr(m[1], newname).__doc__)
                    

    unit_conversion_decorator(Mirheo)  # Decorate all methods.
    Mirheo.__init__ = decorate_coordinator(Mirheo.__init__)
    Mirheo.registerPlugins = decorate_func_with_plugin_arg(Mirheo.registerPlugins)
    Mirheo.deregisterPlugins = decorate_func_with_plugin_arg(Mirheo.deregisterPlugins)


__init__()
