import inspect
import torch
import collections
import types
import textwrap
import functools
import warnings

import torch._jit_internal as _jit_internal
from torch.jit.frontend import get_default_args
from torch.nn import Module, ModuleList, Sequential, ModuleDict
from torch._six import get_function_from_type, bind_method


ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))

# TODO: there should be a more principled way of doing this.
blacklist = [
    "_version",
    "_parameters",
    "_buffers",
    "_modules",
    "_initializing",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "dump_patches",
]

def make_stub(func):
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = torch.jit.get_jit_def(func, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)

def make_stub_from_method(nn_module, method):
    func = get_function_from_type(type(nn_module), method)
    if isinstance(func, ScriptMethodStub):
        return func
    return make_stub(func)

# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, str, type(None), types.FunctionType, torch.device, torch.layout, torch.dtype)

def _get_valid_constant(attr, v):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, tuple) or isinstance(v, list):
        return tuple(_get_valid_constant(attr, x) for x in v)
    constants = ", ".join(typ.__name__ for typ in _constant_types)
    raise TypeError(textwrap.dedent("""
        '{}' object for attribute '{}' is not a valid constant.
        Valid constants are:
        1. a nn.ModuleList
        2. a value of type {{{}}}
        3. a list or tuple of (2)
        """.format(type(v).__name__, attr, constants)))

def infer_raw_concrete_type(nn_module):
    """
    Build a ConcreteModuleType from an nn.Module. This ConcreteModuleType
    doesn't have a JIT type associated with it yet, it must be filled in
    by the caller.
    """
    concrete_type = torch._C.ConcreteModuleType()
    concrete_type.add_pyclass(type(nn_module))
    if isinstance(nn_module, (torch.nn.ModuleDict, torch.jit._ConstModuleDict)):
        concrete_type.set_module_dict()
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.jit._ConstModuleList)):
        concrete_type.set_module_list()

    class_annotations = getattr(nn_module, '__annotations__', {})

    # try to infer the type from type annotation or from the object itself
    def infer_type(name, item):
        if name in class_annotations:
            attr_type = torch.jit.annotations.ann_to_type(class_annotations[name])
        elif isinstance(item, torch.jit.Attribute):
            attr_type = torch.jit.annotations.ann_to_type(item.type)
        else:
            attr_type = torch._C._jit_try_infer_type(item)
        return attr_type

    added_names = set()

    for name, item in nn_module._parameters.items():
        if item is None:
            # TODO special case: parameters can be None. The JIT assumes
            # parameters are Tensor types, so in this case just add it as a
            # attribute.
            # The "correct" fix here is to add the parameter as a NoneType
            # attribute, but NoneType refinemenet is currently wonky
            continue
        assert isinstance(item, torch.Tensor)
        attr_type = infer_type(name, item)
        concrete_type.add_attribute(name, attr_type, True)
        added_names.add(name)

    for name, item in nn_module._buffers.items():
        if item is None:
            # TODO special case: parameters can be None. The JIT assumes
            # parameters are Tensor types, so in this case just add it as a
            # attribute.
            # The "correct" fix here is to add the parameter as a NoneType
            # attribute, but NoneType refinemenet is currently wonky
            continue
        assert isinstance(item, torch.Tensor)
        attr_type = infer_type(name, item)
        concrete_type.add_attribute(name, attr_type, False)
        added_names.add(name)

    for name, item in nn_module._modules.items():
        attr_type = infer_type(name, item)
        if attr_type is not None:
            # if the type can be inferred, it should be a module interface type
            concrete_type.add_module_interface(name, attr_type)
        else:
            # otherwise we get the concrete module type for item and add it to concrete_type
            sub_concrete_type = concrete_type_store.get_or_create_concrete_type(item)
            concrete_type.add_module(name, sub_concrete_type)

        added_names.add(name)

    # populate constants_set
    constants_set = getattr(nn_module, "__constants__", set())

    # Constants annotated via `Final[T]` rather than being added to `__constants__`
    for name, ann in class_annotations.items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    for name in constants_set:
        if name in added_names:
            # XXX: It is possible for something to be in the constants set but
            # also in the parameters/buffers. This happens in BatchNorm as a
            # hack to support optional parameters.
            continue
        if not hasattr(nn_module, name):
            # TODO: We should really error in this case, but there are a couple
            # extant examples of this so leave it for a future PR.
            warnings.warn("'{}' was found in ScriptModule constants, "
                          "but was not actually set in __init__. "
                          "Consider removing it.".format(name))
            continue
        value = getattr(nn_module, name)
        concrete_type.add_constant(name, _get_valid_constant(name, value))
        added_names.add(name)

    # populate overloads
    overloads = getattr(nn_module, "__overloads__", {})
    # update with any annotated overloads
    overloads.update(get_overload_name_mapping(get_overload_annotations(nn_module)))
    for name, overloaded_names in overloads.items():
        concrete_type.add_overload(name, overloaded_names)

    # TODO: [switch to __dict__]
    # we should use __dict__ here because we only want to pick up attributes on
    # this module instance, not the class itself. We can't do it right now
    # because there is code that relies on properties being turned into attributes.
    # This is wrong (the property function is only evaluated once then "saved"
    # as an attribute), so we should fix that and then switch this to using __dict__
    for name in dir(nn_module):
        if name in blacklist or name.startswith("__"):
            # Python objects have lots of random attributes attached to them;
            # PyTorch adds a few more. Prevent these from getting compiled.
            continue

        if name in added_names:
            # Don't re-add anything we already added
            continue

        if not hasattr(nn_module, name):
            # TODO: delete this when [switch to __dict__]
            continue

        item = getattr(nn_module, name)
        if name not in nn_module.__dict__ and not isinstance(getattr(type(nn_module), name, None), property):
            # Skip class attributes that aren't properties
            # TODO: delete this when [switch to __dict__]
            continue

        # Handle Python function attributes
        if inspect.isfunction(item) and not inspect.ismethod(item):
            cls_attr = getattr(type(nn_module), name, None)
            if inspect.isfunction(cls_attr):
                # Skip function attributes that exist on the nn_module class.
                # TODO: delete this when [switch to __dict__]
                continue

            try:
                scripted_fn = torch.jit.script(item)
                concrete_type.add_function_attribute(
                    name,
                    torch._C._jit_try_infer_type(scripted_fn),
                    item)
            except Exception as e:
                # If we fail to script the function, it isn't a hard error.
                # Instead, we will add it to the list of attributes we failed
                # to convert, with the compilation error.
                hint = ("(This function exists as an attribute on the Python module, "
                        "but we failed to compile it to a TorchScript function. "
                        "\nThe error stack is reproduced here:\n{}").format(e)
                concrete_type.add_failed_attribute(name, hint)
                pass

            continue

        # Handle Script function attributes
        if isinstance(item, torch.jit.ScriptFunction):
            concrete_type.add_function_attribute(
                name,
                torch._C._jit_try_infer_type(item),
                item)
            continue

        # If we got here, this is a regular "data" attribute, Add it to the concrete type
        attr_type = infer_type(name, item)
        if attr_type is not None:
            concrete_type.add_attribute(name, attr_type, False)
        else:
            # TODO: could add more detail here. For example, what the user should do
            # when the pytype is `list` or `NoneType`
            hint = ("(This attribute exists on the Python module, "
                    "but we failed to convert Python type: '{}' "
                    "to a TorchScript type.)").format(type(item).__name__)
            concrete_type.add_failed_attribute(name, hint)

    return concrete_type

class ConcreteTypeStore(object):
    def __init__(self):
        # Python module type => List[ConcreteModuleType)]
        self.type_store = {}
        # ConcreteTypes that have had their methods already compiled
        self.methods_compiled = set()

    def get_or_create_concrete_type(self, nn_module):
        """
        Infer a ConcreteType from this `nn.Module` instance. Underlying JIT
        types are re-used if possible.
        """
        assert isinstance(nn_module, Module)
        if isinstance(nn_module, torch.jit.ScriptModule) and \
                hasattr(nn_module, "_concrete_type"):
            return nn_module._concrete_type

        if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)):
            # TODO: This is here because the compilation path for constant iterable
            # modules is different from everything else. Instead of calling
            # create_script_module, we directly create a
            # _ConstSequential/ModuleList/ModuleDict instance.
            #
            # The path used to create ConcreteTypes involves going in and analyzing
            # all the nn.Modules ahead of time.
            #
            # That leads to skew where the result of generating a ConcreteType
            # (which involves looking at torch.nn.Sequential) is different from the
            # actual compilation path (which directly builds _ConstSequential).
            #
            # The right solution is to make these modules not special in the
            # compilation path. But for now, just mimic what compilation does when
            # generating a ConcreteType
            scripted = create_constant_iterable_module(nn_module)
            return scripted._concrete_type

        raw_concrete_type = infer_raw_concrete_type(nn_module)

        nn_module_type = type(nn_module)
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []

        # Search the type store for an already-available JIT type
        known_types = self.type_store[nn_module_type]
        found = False
        for known_type in known_types:
            if raw_concrete_type.equals(known_type):
                return known_type

        # We didn't find anything; generate a new JIT type from this concrete type
        raw_concrete_type.create_new_type_from_this()
        self.type_store[nn_module_type].append(raw_concrete_type)
        return raw_concrete_type

concrete_type_store = ConcreteTypeStore()

def create_methods_from_stubs(concrete_type, stubs):
    defs = [m.def_ for m in stubs]
    name = [def_.name().name for def_ in defs]
    rcbs = [m.resolution_callback for m in stubs]
    defaults = [get_default_args(m.original_method) for m in stubs]
    concrete_type._create_methods(defs, rcbs, defaults)

def create_script_module_for_tracing(nn_module, stubs):
    """
    Creates a new ScriptModule from an nn.Module, but always uses a fresh type.

    NOTE: Only use this when we cannot guarantee type sharing will work
          correctly. This only happens today for traced modules, where the same
          module can produce different traced methods depending on the inputs.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs:  ScriptMethodStubs to compile as part of the conversion process.
    """
    check_module_initialized(nn_module)
    # Get a ConcreteType without a JIT type. We will generate one ourselves
    # and fill it in.
    concrete_type = infer_raw_concrete_type(nn_module)
    cpp_module = torch._C.ScriptModule(torch._jit_internal._qualified_name(type(nn_module)),
                                       torch.jit._python_cu,
                                       True)
    # Poison this concrete type to ensure that it never gets re-used
    concrete_type.set_poisoned()
    concrete_type.add_jit_type(cpp_module._type())

    return create_script_module_impl(nn_module, concrete_type, cpp_module, stubs)


def create_script_module(nn_module, stubs):
    """
    Creates a new ScriptModule from an nn.Module, sharing underlying JIT types if possible

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs:  ScriptMethodStubs to compile as part of the conversion process.
    """
    check_module_initialized(nn_module)
    concrete_type = concrete_type_store.get_or_create_concrete_type(nn_module)
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)

    return create_script_module_impl(nn_module, concrete_type, cpp_module, stubs)

def create_script_module_impl(nn_module, concrete_type, cpp_module, stubs):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        cpp_module:  A newly-constructed C++ script::Module to copy stuff into.
        stubs:  ScriptMethodStubs to compile as part of the conversion process.
    """
    assert concrete_type.jit_type and concrete_type.jit_type == cpp_module._type()

    def init_fn(script_module):
        # Initialize the ScriptModule:
        # 1. Copy the attributes/parameters/buffers from the original `nn_module` to the new ScriptModule.
        for name, (attr_type, is_param) in concrete_type.get_attributes().items():
            orig_value = getattr(nn_module, name)
            if is_param:
                cpp_module._register_parameter(name, orig_value, False)
            else:
                orig_value = orig_value.value if isinstance(orig_value, torch.jit.Attribute) else orig_value
                cpp_module._register_attribute(name, attr_type, orig_value)

        # 2. Copy the submodules from the original `nn_module` to the new ScriptModule,
        #    recursively scripting them.
        for name, module_type in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            assert isinstance(orig_value, Module)
            if isinstance(module_type, torch._C.InterfaceType):
                # use the interface inference rule to compile the module
                scripted = interface_script(module_type, orig_value)
            else:
                # use the default recursive rule to compile the module
                scripted = recursive_script(orig_value)
            cpp_module._register_module(name, scripted._c)

            script_module._modules[name] = scripted

        # 3. Copy @ignored/@unused methods from the original `nn_module` to the new ScriptModule.
        #    This ensures we can access these Python methods on the ScriptModule.
        for name in dir(nn_module):
            item = getattr(nn_module, name, None)
            if not inspect.ismethod(item):
                continue
            if _jit_internal.is_ignored_fn(item):
                setattr(script_module, name, item)

        # For convenience, attach the concrete type to the new ScriptModule
        script_module._concrete_type = concrete_type

    # Actually create the ScriptModule, initializing it with the function we just defined
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

    # Compile methods if necessary
    if concrete_type not in concrete_type_store.methods_compiled:
        create_methods_from_stubs(concrete_type, stubs)
        torch._C._run_emit_module_hook(cpp_module)
        concrete_type_store.methods_compiled.add(concrete_type)

    # Make the compiled methods available to the Python ScriptModule class.
    for stub in stubs:
        if stub.original_method is None:
            # define()'d methods don't have an Python original_method, so we
            # don't need to do any Python re-wrapping stuff
            continue

        name = stub.original_method.__name__
        if name != stub.def_.name().name:
            # TODO: Why skip this? Because @torch.jit._overload_method will
            # mangle the name of the function.
            continue
        script_method = cpp_module._get_method(name)

        # Wrap the original to propagate docstrings and such.
        # TODO: we don't currently do this functions that are recursively
        # compiled, we should.
        script_method = functools.wraps(stub.original_method)(script_method)

        # Add the methods to the script_module directly. This ensures they will
        # be found first when `name` is looked up (as opposed to the stubs or
        # nn.Module.forward)
        script_module.__dict__[name] = script_method

    return script_module

def get_overload_annotations(mod):
    # original function => [(mangled overload name, overload function)]
    overloads = {}
    for name in dir(mod):
        item = getattr(mod, name, None)
        if not callable(item):
            continue

        # builtin functions like repr() in python 2 do not have __module__ defined
        if hasattr(item, "__module__") and item.__module__ is not None:
            method_overloads = _jit_internal._get_overloaded_methods(item, mod.__class__)
            if method_overloads is None:
                continue

            original_name = item.__name__
            names = [name + "__" + str(i) for i in range(len(method_overloads))]
            overloads[item] = list(zip(names, method_overloads))

    return overloads

def get_overload_name_mapping(overload_info):
    # Same format as __overloads__
    # original function => [overload names]
    overload_name_mappings = {}
    for orig_fn, overloads in overload_info.items():
        original_name = orig_fn.__name__
        if original_name not in overload_name_mappings:
            overload_name_mappings[original_name] = []

        for overload_name, _ in overloads:
            overload_name_mappings[original_name].append(overload_name)
    return overload_name_mappings

def make_stubs_for_overloads(overload_info):
    overload_stubs = []
    for orig_fn, overloads in overload_info.items():
        orig_ast = torch.jit.get_jit_def(orig_fn, self_name="RecursiveScriptModule")
        for overload_name, overload_fn in overloads:
            torch.jit._check_no_signature(overload_fn)
            over_ast = torch.jit.get_jit_def(overload_fn, self_name="RecursiveScriptModule")
            new_ast = torch._C._replace_overloaded_method_decl(over_ast.decl(), orig_ast, overload_name)
            _rcb = _jit_internal.createResolutionCallbackFromClosure(orig_fn)
            overload_stubs.append(ScriptMethodStub(_rcb, new_ast, overload_fn))
    return overload_stubs

def check_module_initialized(mod):
    assert isinstance(mod, torch.nn.Module)
    if not hasattr(mod, '_parameters'):
        raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                           .format(type(mod).__name__))

def infer_methods_to_compile(nn_module):
    """
    Implements the default rules for which methods should act as starting
    points for compilation (TODO add a link when the rules are published).
    """
    check_module_initialized(nn_module)

    methods = []
    if hasattr(nn_module, 'forward'):
        if getattr(nn_module.forward, "__func__", None) == torch.nn.Module.forward:
            # TODO, we deleted a check that forward is actually defined, instead skipping it
            pass
        elif not _jit_internal.is_ignored_fn(nn_module.forward):
            methods = ['forward']

    exported = []
    for name in dir(nn_module):
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            exported.append(name)

    methods = methods + exported

    overload_name_mappings = dict(getattr(nn_module, "__overloads__", {}))
    overload_info = get_overload_annotations(nn_module)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)

    nn_module.__overloads__ = overload_name_mappings

    # we shouldn't directly compile overloaded methods, just its overloads
    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings

    filtered_methods = filter(ignore_overloaded, methods)

    # Unique the methods. We don't want to use a set to store the methods because it
    # introduces non-determinism to compile order.
    uniquer = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)

    stubs = []
    for method in uniqued_methods:
        stubs.append(make_stub_from_method(nn_module, method))
    return overload_stubs + stubs

def infer_interface_methods_to_compile(mod_interface, nn_module):
    """
    Rule to infer the methods from the interface type to know which
    methods need to act as starting points for compilation.
    """
    stubs = []
    for method in mod_interface.getMethodNames():
        stubs.append(make_stub_from_method(nn_module, method))
    return stubs

def interface_script(mod_interface, nn_module):
    """
    Makes a ScriptModule from an nn.Module, using the interface methods rule for
    determining which methods to compile.

    Arguments:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module

    check_module_initialized(nn_module)
    return create_script_module(nn_module, infer_interface_methods_to_compile(mod_interface, nn_module))

def recursive_script(nn_module):
    """
    Makes a ScriptModule from an nn.Module, using the default rules for
    determining which methods to compile.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module

    check_module_initialized(nn_module)

    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)):
        # Create constant versions for the iterable modules
        return create_constant_iterable_module(nn_module)

    return create_script_module(nn_module, infer_methods_to_compile(nn_module))

def try_compile_fn(fn, loc):
    if _jit_internal.is_ignored_fn(fn):
        # Don't do anything for @ignore'd functions
        return None

    if isinstance(fn, torch.nn.Module):
        # Since modules are callable pybind recognizes them as functions, but
        # don't do anything for them
        return None

    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        raise RuntimeError("`{}` is not a function. Recursive scripting only supports "
                           "Python functions or methods currently.\n"
                           "Consider manually annotating `{}` with @torch.jit.script.".format(fn, fn))

    # We don't have the actual scope where the function was defined, but we can
    # extract the necessary info from the closed over variables on the function
    # object
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    return torch.jit.script(fn, _rcb=rcb)

def create_constant_iterable_module(module):
    modules = collections.OrderedDict()

    for key, submodule in module._modules.items():
        if isinstance(submodule, (ModuleList, Sequential, ModuleDict)):
            # Make each item in the module a constant
            modules[key] = create_constant_iterable_module(submodule)
        else:
            modules[key] = recursive_script(submodule)

    if isinstance(module, Sequential):
        return torch.jit._ConstSequential(Sequential(modules))
    elif isinstance(module, ModuleList):
        return torch.jit._ConstModuleList(modules)
    elif isinstance(module, ModuleDict):
        return torch.jit._ConstModuleDict(modules)
    else:
        raise RuntimeError("Only nn.ModuleList, nn.Sequential, and nn.ModuleDict can be made "
                           "into constant modules, found {}".format(module))

def wrap_cpp_module(cpp_module):
    """
    Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules
    """
    def init_fn(script_module):
        for name, cpp_module in script_module._c._get_modules():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
    return torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

def compile_unbound_method(concrete_type, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    stub = make_stub(fn)
    with torch.jit._disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        create_methods_from_stubs(concrete_type, (stub,))
    return stub

def lazy_bind(concrete_type, unbound_method):
    """
    Returns a function that lazily binds `unbound_method` to a provided
    Module IValue, then invokes the method. We do this so that any Python
    shenanigans that will poison type sharing are impossible at compile
    time.
    """
    def lazy_binding_method(cpp_module, *args):
        def init_fn(script_module):
            orig_class = concrete_type.py_class

            # Copy @ignored/@unused methods from the original module to the new one.
            # This ensures they are available during execution.
            for name in dir(orig_class):
                item = getattr(orig_class, name, None)
                if _jit_internal.is_ignored_fn(item):
                    setattr(script_module, name, item)

            # Copy constants over so they are available during execution.
            for name, value in concrete_type.get_constants().items():
                setattr(script_module, name, value)

        script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
        method = bind_method(unbound_method, script_module, torch.jit.RecursiveScriptModule)
        return method(*args)

    # make the lazy binding method "look like" the original method
    lazy_binding_method.original_fn = unbound_method
    lazy_binding_method.__name__ = unbound_method.__name__
    torch._jit_internal.copy_torchscript_modifier(unbound_method, lazy_binding_method)

    return lazy_binding_method
