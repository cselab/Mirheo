#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:57:08 2018

@author: alexeedm
"""

import lxml.etree as ET
import inspect
import typing
from functools import wraps
import traceback


###########################################################################################################################
#//////|   Decorator   |//////////////////////////////////////////////////////////////////////////////////////////////////#
# https://stackoverflow.com/questions/1389180/automatically-initialize-instance-variables                                 #
###########################################################################################################################

def initializer_with_name_and_parent(superclass):
    
    def outer(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            self.name = _util_get_my_name(self, 3)
            superclass.__init__(self)
            
            
            _assign_args(self, list(args), kwargs, function)
            function(self, *args, **kwargs)
    
        return wrapped
    return outer
    

def initializer(function):

  @wraps(function)
  def wrapped(self, *args, **kwargs):
    _assign_args(self, list(args), kwargs, function)
    function(self, *args, **kwargs)

  return wrapped

###########################################################################################################################
#//////|   Utils   |//////////////////////////////////////////////////////////////////////////////////////////////////////#
###########################################################################################################################

def _assign_args(instance, args, kwargs, function):

  def set_attribute(instance, parameter, default_arg):
    if not(parameter.startswith("_")):
      setattr(instance, parameter, default_arg)

  def assign_keyword_defaults(parameters, defaults):
    for parameter, default_arg in zip(reversed(parameters), reversed(defaults)):
      set_attribute(instance, parameter, default_arg)

  def assign_positional_args(parameters, args):
    for parameter, arg in zip(parameters, args.copy()):
      set_attribute(instance, parameter, arg)
      args.remove(arg)

  def assign_keyword_args(kwargs):
    for parameter, arg in kwargs.items():
      set_attribute(instance, parameter, arg)
  def assign_keyword_only_defaults(defaults):
    return assign_keyword_args(defaults)

  def assign_variable_args(parameter, args):
    set_attribute(instance, parameter, args)

  POSITIONAL_PARAMS, VARIABLE_PARAM, _, KEYWORD_DEFAULTS, _, KEYWORD_ONLY_DEFAULTS, _ = inspect.getfullargspec(function)
  POSITIONAL_PARAMS = POSITIONAL_PARAMS[1:] # remove 'self'

  if(KEYWORD_DEFAULTS     ): assign_keyword_defaults     (parameters=POSITIONAL_PARAMS,  defaults=KEYWORD_DEFAULTS)
  if(KEYWORD_ONLY_DEFAULTS): assign_keyword_only_defaults(defaults=KEYWORD_ONLY_DEFAULTS                          )
  if(args                 ): assign_positional_args      (parameters=POSITIONAL_PARAMS,  args=args                )
  if(kwargs               ): assign_keyword_args         (kwargs=kwargs                                           )
  if(VARIABLE_PARAM       ): assign_variable_args        (parameter=VARIABLE_PARAM,      args=args                )


def actual_kwargs():
    """
    Decorator that provides the wrapped function with an attribute 'actual_kwargs'
    containing just those keyword arguments actually passed in to the function.
    """
    def decorator(function):
        def inner(*args, **kwargs):
            inner.actual_kwargs = kwargs
            return function(*args, **kwargs)
        return inner
    return decorator

###########################################################################################################################

int3 = typing.Tuple[int, int, int]
float3 = typing.Tuple[float, float, float]

def assert_str(val):
    assert isinstance(val, str)
    assert len(val) > 0
    

def assert_bool(val):
    assert isinstance(val, bool)
    
    
def assert_int(val):
    assert isinstance(val, int)
    
def assert_posint(val):
    assert_int(val)
    assert val > 0
    
def assert_nonnegint(val):
    assert_int(val)
    assert val >= 0
    
    
def assert_float(val):
    assert isinstance(val, int) or isinstance(val, float)
    
def assert_posfloat(val):
    assert_float(val)
    assert val > 0.0

def assert_nonnegfloat(val):
    assert_float(val)
    assert val > -1e-20
    
def assert_posfloat_or_none(val):
    if val is None:
        return
    assert_posfloat(val)
    
def assert_int3(val):
    assert isinstance(val, tuple) and len(val) == 3
    for v in val:
        assert_int(v)
    
def assert_posint3(val):
    assert_int3(val)
    assert val[0] > 0 and val[1] > 0 and val[2] > 0
    
    
def assert_float3(val):
    assert isinstance(val, tuple) and len(val) == 3
    for v in val:
        assert_float(v)
    
def assert_posfloat3(val):
    assert_float3(val)
    assert val[0] > 0.0 and val[1] > 0.0 and val[2] > 0.0
        
def assert_nonnegfloat3(val):
    assert_float3(val)
    assert val[0] > -1e-20 and val[1] > -1e-20 and val[2] > -1e-20
    

def assert_direction(val):
    assert val == 'x' or val == 'y' or val == 'z'


###########################################################################################################################

def _util_to_string(arg):
    if isinstance(arg, tuple) or isinstance(arg, list):
        return ' '.join( [str(v) for v in arg] )
    else:
        return str(arg)

def _util_get_user_attributes(cls):
    return [ (item, getattr(cls, item)) for item in cls._properties_.keys() ]

def _util_toxml(c, name : str):
    node = ET.Element(name)
    set_properties = _util_get_user_attributes(c)
    
    node.set('type', c.mytype)
    for a in set_properties:
        if not(a[1] is None):
            node.set(a[0], _util_to_string(a[1]))
        
    return node


# https://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
def _util_get_my_name(cls, depth = 2):
    (filename,line_number,function_name,text)=traceback.extract_stack()[-depth]
    name = text[:text.find('=')].strip()
    
    if name.find(',') < 0:
        return name
    else:
        raise NameError('Could not retrieve object name')
        
def _util_validate_list(cls, lst):
    for name, func in lst:
        try:
            func(getattr(cls, name))
        except:
            raise TypeError('Wrong variable type: "' + name + '"')


###########################################################################################################################
# Particle Vectors
###########################################################################################################################

class ParticleVector:
        
    def set_properties(self):
        self._properties_ = {}
    
    def validate(self):
        self.set_properties()
        _util_validate_list(self, self._properties_.items())

        assert isinstance(self.ic, InitialCondition)
        self.ic.validateParticleVector(self)
    
    def _toxml(self):
        self.validate()
        icn = self.ic._toxml()
        pvn = _util_toxml(self, 'particle_vector')
        pvn.append(icn)
        
        return pvn

class ObjectVector:
    pass
    
class SimplePV(ParticleVector):
    mytype = 'regular'
       
    @initializer_with_name_and_parent(ParticleVector)
    def __init__(self, ic, mass : float = 1.0, checkpoint_every : int = 0):
        self.validate()

        
    def set_properties(self):
        ParticleVector.set_properties(self)
        self._properties_.update({'name'             : assert_str,
                                  'mass'             : assert_posfloat,
                                  'checkpoint_every' : assert_nonnegint})
        
            
class Membrane(SimplePV, ObjectVector):
    mytype = 'membrane'
    
    @initializer_with_name_and_parent(ParticleVector)
    def __init__(self, ic, mass : float = 1.0, checkpoint_every : int = 0,
                 particles_per_object : int = 0, mesh_filename : str = "mesh.off"):
        self.validate()
        
    def set_properties(self):
        SimplePV.set_properties(self)
        self._properties_.update({'particles_per_object' : assert_posint,
                                  'mesh_filename'        : assert_str})
        
            
class RigidObject(SimplePV, ObjectVector):
    mytype = 'rigid_objects'

    @initializer_with_name_and_parent(ParticleVector)
    def __init__(self, ic, mass : float = 1.0, checkpoint_every : int = 0,
                 particles_per_object : int = 0, mesh_filename : str = "mesh.off", moment_of_inertia : float3 = (0, 0, 0)):
        self.validate()
        
    def set_properties(self):
        SimplePV.set_properties(self)
        self._properties_.update({'particles_per_object' : assert_posint,
                                  'mesh_filename'        : assert_str,
                                  'moment_of_inertia'    : assert_posfloat3})
        
            
class RigidEllipsoid(SimplePV, ObjectVector):
    mytype = 'rigid_ellipsoids'
    
    @initializer_with_name_and_parent(ParticleVector)
    def __init__(self, ic, mass : float = 1.0, checkpoint_every : int = 0,
                 particles_per_object : int = 0, semi_axes : float3 = (0, 0, 0)):
        self.validate()
        
    def set_properties(self):
        SimplePV.set_properties(self)
        self._properties_.update({'particles_per_object' : assert_posint,
                                  'semi_axes'            : assert_posfloat3})

            
###########################################################################################################################
# Initial Conditions
###########################################################################################################################
    
class InitialCondition:
    
    def set_properties(self):
        self._properties_ = {}
    
    def validate(self):
        self.set_properties()
        _util_validate_list(self, self._properties_.items())
    
    def _toxml(self):
        self.validate()
        return _util_toxml(self, 'generate')
    

class UniformIC(InitialCondition):
    mytype = 'uniform'
    
    @initializer
    def __init__(self, density : float = 1.0):
        self.validate()
        
    def set_properties(self):
        InitialCondition.set_properties(self)
        self._properties_.update({'density' : assert_posfloat})
        
    def validateParticleVector(self, pv):
        assert not isinstance(pv, ObjectVector)
    

class RigidIC(InitialCondition):
    mytype = 'read_rigid'
    
    @initializer
    def __init__(self, ic_filename : str = 'objects.ic', xyz_filename : str = 'object.xyz'):
        self.validate()
        
    def set_properties(self):
        InitialCondition.set_properties(self)
        self._properties_.update({'ic_filename'  : assert_str,
                                  'xyz_filename' : assert_str})
    
    def validateParticleVector(self, pv):
        assert isinstance(pv, RigidObject) or isinstance(pv, RigidEllipsoid)
    

class MembraneIC(InitialCondition):
    mytype = 'read_membranes'
    
    @initializer
    def __init__(self, ic_filename : str = 'membranes.ic', global_scale : float = 1.0):
        self.validate()
        
    def set_properties(self):
        InitialCondition.set_properties(self)
        self._properties_.update({'ic_filename'  : assert_str,
                                  'global_scale' : assert_posfloat})
        
    def validateParticleVector(self, pv):
        assert isinstance(pv, Membrane)
    

class RestartIC(InitialCondition):
    mytype = 'restart'
    
    @initializer
    def __init__(self, path : str = 'restart/'):
        self.validate()
        
    def set_properties(self):
        InitialCondition.set_properties(self)
        self._properties_.update({'path'  : assert_str})
    
    def validateParticleVector(self, pv):
        pass
    
###########################################################################################################################
# Interactions
###########################################################################################################################

class Interaction:
    
    def __init__(self):
        self.particle_vectors = []
        
    def set_properties(self):
        self._properties_ = {}
    
    def validate(self):
        self.set_properties()
        _util_validate_list(self, self._properties_.items())
    
    def _toxml(self):
        self.validate()
        node = _util_toxml(self, 'interaction')
        for pair in self.particle_vectors:
            pv1 = pair[0][0]
            pv2 = pair[0][1]
            
            applyto = ET.Element('apply_to', pv1=pv1.name, pv2=pv2.name)
            for attr, val in pair[1:]:
                if not(val is None):
                    applyto.set(attr, _util_to_string(val))
            node.append(applyto)
        
        return node
    
class DPD(Interaction):
    mytype = 'dpd'
    
    @initializer_with_name_and_parent(Interaction)
    def __init__(self,
                 rc    : float = 1.0,
                 a     : float = 10.0,
                 gamma : float = 10.0,
                 kbt   : float = 1.0,
                 dt    : float = 0.01,
                 power : float = 1.0,
                 
                 stress_period : float = None):
        self.validate()
        
    def set_properties(self):
        Interaction.set_properties(self)
        self._properties_.update({'rc'    : assert_posfloat,
                                  'a'     : assert_posfloat,
                                  'gamma' : assert_posfloat,
                                  'kbt'   : assert_posfloat,
                                  'dt'    : assert_posfloat,
                                  'power' : assert_posfloat})
        
    
    @actual_kwargs()
    def addParticleVectors(self, pv1, pv2,
                           a     : float = None,
                           gamma : float = None,
                           kbt   : float = None,
                           dt    : float = None,
                           power : float = None):
        
        assert isinstance(pv1, ParticleVector)
        assert isinstance(pv2, ParticleVector)

        lcl = locals()
        prop = self._properties_
        res = ((pv1, pv2),)
        
        for name, value in lcl.items():
            if not (value is None) and name in prop.keys():
                try:
                    prop[name](value)
                except:
                    raise TypeError('Wrong variable type: "' + name + '"')
                    
                res = res + ((name, value),)
        
        self.particle_vectors.append(res)
        
        

class LJ(Interaction):
    mytype = 'lj'
    
    @initializer_with_name_and_parent(Interaction)
    def __init__(self,
                 rc        : float = 1.0,
                 epsilon   : float = 10.0,
                 sigma     : float = 0.5,
                 max_force : float = 1000.0):
        self.validate()
        
    def set_properties(self):
        Interaction.set_properties(self)
        self._properties_.update({'rc'        : assert_posfloat,
                                  'epsilon'   : assert_posfloat,
                                  'sigma'     : assert_posfloat,
                                  'max_force' : assert_posfloat})
        
     
    
    @actual_kwargs()
    def addParticleVectors(self, pv1, pv2,
                           rc        : float = None,
                           epsilon   : float = None,
                           sigma     : float = None,
                           max_force : float = None):
                
        assert isinstance(pv1, ParticleVector)
        assert isinstance(pv2, ParticleVector)
        
        lcl = locals()
        prop = self._properties_
        res = ((pv1, pv2),)
        
        for name, value in lcl.items():
            if not (value is None) and name in prop.keys():
                try:
                    prop[name](value)
                except:
                    raise TypeError('Wrong variable type: "' + name + '"')
                    
                res = res + ((name, value),)
        
        self.particle_vectors.append(res)


class LJ_ObjectAware(LJ):
    mytype = 'lj_object'
    
    def __init__(self, *args, **kwargs):
        self.name = _util_get_my_name(self)
        try:
            super(LJ_ObjectAware, self).__init__(*args, **kwargs)
        except:
            pass
        
    def addParticleVectors(self, *args, **kwargs):
        super(LJ_ObjectAware, self).addParticleVectors(*args, **kwargs)
        assert isinstance(args[1], ObjectVector) or isinstance(args[2], ObjectVector)
        

class MembraneForces(Interaction):
    mytype = 'membrane'
    
    @initializer_with_name_and_parent(Interaction)
    def __init__(self,
                 stress_free : bool = True,
                 grow_until : float = None,
                 **kwargs):
        
        self.validate()
        self.rc = 1.0

    def set_properties(self):
        Interaction.set_properties(self)
        self._properties_.update({'stress_free' : assert_bool,
                                  'grow_until'  : assert_posfloat_or_none,
                                  'x0'          : assert_nonnegfloat,
                                  'p'           : assert_nonnegfloat,
                                  'ka'          : assert_nonnegfloat,
                                  'kb'          : assert_nonnegfloat,
                                  'kd'          : assert_nonnegfloat,
                                  'kv'          : assert_nonnegfloat,
                                  'gammaC'      : assert_nonnegfloat,
                                  'gammaT'      : assert_nonnegfloat,
                                  'kbt'         : assert_nonnegfloat,
                                  'mpow'        : assert_nonnegfloat,
                                  'theta'       : assert_nonnegfloat,
                                  'area'        : assert_nonnegfloat,
                                  'volume'      : assert_nonnegfloat})
    
    @actual_kwargs()
    def addParticleVector(self, pv):
        assert isinstance(pv, Membrane)
        res = ((pv, pv),)
        self.particle_vectors.append(res)

###########################################################################################################################
# Integrators
###########################################################################################################################     
 
class Integrator:
    
    def __init__(self):
        self.particle_vectors = []
        
    def set_properties(self):
        self._properties_ = {}
    
    def validate(self):
        self.set_properties()
        _util_validate_list(self, self._properties_.items())
       
    def _toxml(self):
        self.validate()
        node = _util_toxml(self, 'integrator')
        for pv in self.particle_vectors:
            applyto = ET.Element('apply_to', pv=pv.name)
            node.append(applyto)
            
        return node
    
    def addParticleVector(self, pv):
        self.particle_vectors.append(pv)
        

class VelocityVerlet(Integrator):
    mytype='vv'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt' : assert_nonnegfloat})

class VelocityVerlet_constPressure(Integrator):
    mytype='vv_const_dp'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01, extra_force : float3 = (0, 0, 0)):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt'          : assert_nonnegfloat,
                                  'extra_force' : assert_float3})        
    
class VelocityVerlet_periodicPoiseuille(Integrator):
    mytype='vv_periodic_poiseuille'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01, direction : str = 'x', force : float = 0.0):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt'        : assert_nonnegfloat,
                                  'force'     : assert_float3,
                                  'direction' : assert_direction }) 
        
        
class Rigid_VelocityVerlet(Integrator):
    mytype='vv_rigid'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt' : assert_nonnegfloat})
        
    def addParticleVector(self, pv):
        assert isinstance(pv, ObjectVector)
        self.particle_vectors.append(pv)
        
        
class Move_constVelocity(Integrator):
    mytype='translate'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01, velocity : float3 = (0, 0, 0)):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt'       : assert_nonnegfloat,
                                  'velocity' : assert_float3})
        

class Move_constAngularVelocity(Integrator):
    mytype='const_omega'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01, omega : float3 = (0, 0, 0), center : float3 = (0, 0, 0)):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt'     : assert_nonnegfloat,
                                  'omega'  : assert_float3,
                                  'center' : assert_float3})        

class Move_oscillate(Integrator):
    mytype='oscillate'
    
    @initializer_with_name_and_parent(Integrator)
    def __init__(self, dt : float = 0.01, velocity : float3 = (0, 0, 0), period : float = 0.0):
        self.validate()
        
    def set_properties(self):
        Integrator.set_properties(self)
        self._properties_.update({'dt'       : assert_nonnegfloat,
                                  'velocity' : assert_float3,
                                  'period'   : assert_posfloat})

###########################################################################################################################
# Walls
########################################################################################################################### 

#class Wall(ClassToXML):
#    pass
#    
#class Bouncer(ClassToXML):
#    pass
#
#class Belonger(ClassToXML):
#    pass
#
#class Plugin(ClassToXML):
#    pass

class Simulation:
    
    particle_vectors = []
    belongers = []
    others   = []
    
    @initializer
    def __init__(self, domain : float3, ranks : int3, debug_lvl : int, log_filename : str = 'log'):
        self.validate()
        
    def validate(self):
        assert_posfloat3(self.domain)
        assert_posint3  (self.ranks)
        assert_int(self.debug_lvl)
        assert_str(self.log_filename)
        
    def addParticleVector(self, pv):
        assert isinstance(pv, ParticleVector) and type(pv) != ParticleVector
        self.particle_vectors.append(pv) 
        
    def addInteraction(self, interaction):
        assert isinstance(interaction, Interaction) and type(interaction) != Interaction
        self.others.append(interaction)
        
    def addIntegrator(self, integrator):
        assert isinstance(integrator, Integrator) and type(integrator) != Integrator
        self.others.append(integrator)
        
    def generate(self, iterations : int, filename : str):
        
        root = ET.Element('simulation')
        root.set('name',     'uDeviceX')
        root.set('mpi_ranks', _util_to_string(self.ranks))
        root.set('logfile',   _util_to_string(self.log_filename))
        root.set('debug_lvl', _util_to_string(self.debug_lvl))        
               
        domain = ET.Element('domain', size   = _util_to_string(self.domain))
        run    = ET.Element('run',    niters = _util_to_string(iterations))

        root.append(domain)
        
        for item in self.particle_vectors + self.belongers + self.others:
            root.append(item._toxml())
        
        root.append(run)
        
        tree = ET.ElementTree(root)
        tree.write(filename, pretty_print=True)
        
        return str( ET.tostring(tree, pretty_print=True), 'utf-8' )













