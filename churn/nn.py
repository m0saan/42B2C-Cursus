__all__ = ['Module', 'Linear', 'ReLU', 'CrossEntropyLoss', 'Softmax', 'Dataset', 'DataLoader' ,'SGD']

from tensor import *
import numpy as np
import math
from random import random
import pickle

class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    """

def _unpack_params(value: object):
    """
    Unpack parameters from different Python objects.
    """
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return list(value.parameters())
    elif isinstance(value, dict):
        return [item for v in value.values() for item in _unpack_params(v)]
    elif isinstance(value, (list, tuple)):
        return [item for v in value for item in _unpack_params(v)]
    return []

def _child_modules(value: object):
    """
    Recursively unpack child modules from different Python objects.

    This function takes an object of type `Module`, `dict`, `list`, or `tuple` and 
    recursively extracts any contained `Module` instances, returning them as a list. 
    For other object types, it returns an empty list.
    """
    if isinstance(value, Module):
        return [value] + _child_modules(value.__dict__)
    elif isinstance(value, dict):
        return [item for v in value.values() for item in _child_modules(v)]
    elif isinstance(value, (list, tuple)):
        return [item for v in value for item in _child_modules(v)]
    else:
        return []

def zeros(*shape):
    return Tensor(np.zeros(shape))

def kaiming_uniform(fan_in, fan_out):
    """
    Fills the input Tensor with values according to the method described in
    "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a uniform distribution.
    The resulting tensor will have values sampled from uniform distribution in the range [-std, std] where std = sqrt(2 / fan_in).

    Parameters
    ----------
    fan_in : int
        Number of input units in the weight tensor.
    fan_out : int
        Number of output units in the weight tensor.
    nonlinearity : str, optional
        The non-linear function (`nn.functional` name), recommended to use only with 'relu' or 'leaky_relu'. Default is 'relu'.
    """
    def rand(*shape, low=0.0, high=1.0):
        array = np.random.rand(*shape) * (high - low) + low
        return Tensor(array)

    gain = math.sqrt(2)
    std = gain * math.sqrt(3/fan_in)
    return rand(fan_in, fan_out, low=-std, high=std)


class Module:
    
    def __init__(self):
        self.training = True

    def parameters(self): return _unpack_params(self.__dict__)
    
    def load_weights(self, path):
        with open(path, 'rb') as file:
            loaded_params = pickle.load(file)
        
        for new_p, old_p in zip(loaded_params, self.parameters()):
            old_p.data = new_p.data
            
    def save_weights(self, path):

        # Now you can use pickle to save this dictionary to a file
        with open('model_params.pkl', 'wb') as file:
            pickle.dump(self.parameters(), file)
        
        
    def _children(self): return _child_modules(self.__dict__)
    
    def extra_repr(self) -> str: return ''

    def _get_name(self): return self.__class__.__name__

    def __repr__(self):
        main_str = self._get_name() + '('
        extra_str = self.extra_repr()
        child_str = ''

        for key, module in self.__dict__.items():
            if isinstance(module, Module):
                mod_str = repr(module)
                mod_str = self._add_indent(mod_str, 2)
                child_str += '  (' + key + '): ' + mod_str + '\n'

        if extra_str:
            # If extra information exists, add it to the main string
            main_str += extra_str

        if child_str:
            # If the module has children, add their information to the main string
            main_str += '\n'
            main_str += child_str

        main_str += ')'

        return main_str


    def _add_indent(self, s_, num_spaces):
        """
        Indents each line of the string `s_` with `num_spaces` spaces.
        """
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


    def eval(self):
        """
        Switches the module and all its child modules to evaluation mode.
        """
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        """
        Switches the module and all its child modules to training mode.
        """
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        """
        Defines the call method for the module.
        This method simply calls the forward method and must be overridden by all subclasses.
        """
        self.input = args
        self.output = self.forward(*args, **kwargs)
    
        return self.output


class Linear(Module):
    """
    A class representing a fully connected (linear) layer in a neural network.
    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        weight (Parameter): The weight parameters of the layer.
        bias (Parameter): The bias parameters of the layer, or None if bias=False.
    """
    
    def __init__(self, in_features,  out_features,):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(kaiming_uniform(fan_in=in_features, fan_out=out_features))
        self.bias = Parameter(kaiming_uniform(fan_in=out_features, fan_out=1,).reshape((1, out_features)))
        
    def __repr__(self) -> str:
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'
            
    def forward(self, X):
        
        out = X @ self.weight
        out = out + self.bias.broadcast_to(out.shape) if self.bias else out
        return out


class ReLU(Module):
    def forward(self, x):
        return relu(x)

    
class CrossEntropyLoss(Module):
    def forward(self, input, target):
        def one_hot(num_classes, target):
            return Tensor(np.eye(num_classes)[target])

        log_sum_exp_logits = summation(logsumexp(input, axes=(1, )))
        true_class_logits_sum = summation(input * one_hot(input.shape[1], target.numpy()))
        return (log_sum_exp_logits - true_class_logits_sum) / input.shape[0]
    
class Softmax(Module):
    def forward(self, input: Tensor, dim=1) -> Tensor:
        # import pdb; pdb.set_trace()
        exps = exp(input)
        exps_sum = summation(exps, axes=(1,))
        return exps / broadcast_to(reshape(exps_sum, shape=exps_sum.shape + (1,)), shape=exps.shape)

class Dataset():
    def __getitem__(self, index): raise NotImplementedError
    def __len__(self):raise NotImplementedError

class DataLoader():
    def __init__(self, ds, batch_size): self.ds,self.bs = ds,batch_size
    def __iter__(self):
        for i in range(0, len(self.ds), self.bs): yield self.ds[i:i+self.bs]

class Optimizer:
    def __init__( self, params ): self.params = params
    def step(self): raise NotImplementedError()
    def zero_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(
        self,
        params, # The parameters of the model to be optimized.
        lr=0.01, # The learning rate.
    ):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for self.idx, p in enumerate(self.params):
            self._opt_step(p)

    def _opt_step(self, p):
        grad = Tensor(p.grad, dtype='float32')
        p.data = p.data - grad * self.lr
            
            
class SGDMomentum(Optimizer):
    def __init__(
        self,
        params, # The parameters of the model to be optimized.
        lr=0.01, # The learning rate.
        momentum=0.0, # The momentum factor.
        wd=0.0 # The weight decay (L2 regularization).
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.wd = wd

    def step(self):
        for self.idx, p in enumerate(self.params):
            self._reg_step(p)
            self._opt_step(p)

    def _opt_step(self, p):
        grad = Tensor(p.grad, dtype='float32')
        if self.idx not in self.u:
            self.u[self.idx] = zeros(*p.shape)
        self.u[self.idx] = self.momentum * self.u[self.idx] + (1 - self.momentum) * grad
        p.data = p.data - self.lr * self.u[self.idx]

    def _reg_step(self, p):
        if self.wd != 0:
            p.data *= (1 - self.lr * self.wd)