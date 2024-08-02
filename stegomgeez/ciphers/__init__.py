import importlib
import inspect
import os
from typing_definitions import Callable


__all__ = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.py') and f != '__init__.py']


class Inner:
    pass


class Ciphers:

    def __init__(self):
        self.list = __all__
        self._register_functions()

    def _register_functions(self):
        current_module = inspect.getmodule(self)
        for module_name in self.list:
            inner = Inner()
            module_path = f"{current_module.__name__}.{module_name}"
            module = importlib.import_module(module_path)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, Callable):
                    setattr(inner, attr_name, attr)
            setattr(self, module_name, inner)


ciphers = Ciphers()


def encrypt(cipher_type: str, text: str, key=None):
    """Encrypts text with given cipher type."""
    if cipher_type not in __all__:
        raise LookupError(f'Cipher {cipher_type} not supported')
    return getattr(ciphers, cipher_type)(text, key)


def decrypt(cipher_type: str, text: str, key=None):
    """Decrypts text with given cipher type."""
    if cipher_type not in __all__:
        raise LookupError(f'Cipher {cipher_type} not supported')
    return getattr(ciphers, cipher_type)(text, key)
