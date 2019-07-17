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


# Wrap the __init__ or __new__ method of all the simulation handlers and particle vectors
# If we are not a compute task, just return None
# pass the state if needState is True
def decorate_object(f, needState = True):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        global __coordinator
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


# Wrap the creation of the coordinator
def decorate_coordinator(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        global __coordinator
        f(self, *args, **kwargs)
        
        if __coordinator is not None and  __coordinator() is not None:
           raise Exception('There can only be one coordinator at a time!')
       
        __coordinator = weakref.ref(self)

    return wrapper


# Wrap the registration of the plugins
def decorate_register_plugins(f):
    @functools.wraps(f)
    def wrapper(self, plugins_tuple):
        return f(self, plugins_tuple[0], plugins_tuple[1])

    return wrapper


# Wrap the creation of plugins
# Pass the compute task status into the creation function
# Pass the common global state associated to the coordinator
def decorate_plugins(f):
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
def handle_exception(exc_type, exc_value, exc_traceback):
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if __coordinator is not None and  __coordinator() is not None:
        abort()

def __init__():
    # Setup exception handling
    sys.excepthook = handle_exception
    
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
                    

    # Wrap initialization of the mirheo coordinator
    mirheo.__init__ = decorate_coordinator(mirheo.__init__)
    
    # Wrap registration of the plugins
    mirheo.registerPlugins = decorate_register_plugins(mirheo.registerPlugins)


__init__()
