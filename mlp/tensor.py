from typing import (
    List,
    Optional,
    Tuple,
    Union,
    Set,
)

import numpy
import numpy as ARRAY_API
numpy.set_printoptions(precision=6, linewidth=160)

NDArray = numpy.ndarray
LAZY_MODE = False
TENSOR_COUNTER = 0

class Device:
    """Indicates the device supporting an NDArray."""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "minima.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True
    
    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        return numpy.random.randn(*shape) 

    def rand(self, *shape):
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

def cpu():
    """Return cpu device"""
    return CPUDevice()

def all_devices():
    """return a list of all available devices"""
    return [cpu()]

class Operator:
    def __call__(self, *args):
        raise NotImplementedError()
        
    def compute(self, *args: Tuple[NDArray]):
        raise NotImplementedError()
        
    def gradient(self, out_grad: 'Value', node: 'Value') -> Union['Value', Tuple['Value']]:
        raise NotImplementedError()

class TensorOp(Operator):
    """ Op class specialized to output tensors, will be alternate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

class Value:
    """
    Represents a node within a computational graph.

    This class encapsulates a single value and its relationships in the graph, making it easy to track and manage the value's dependencies, 
    the operation that produced it, and whether it requires a gradient for backpropagation. It's central to the functioning of automatic 
    differentiation within deep learning frameworks.
    """
    op: Optional[Operator]
    children: Set['Value']
    cached_data: NDArray
    requires_grad: bool
    
    def compute_cached_data(self):
        if self.cached_data is None:
            self.cached_data = self.op.compute(*[child.compute_cached_data() for child in self.children])
        return self.cached_data
    
    def is_leaf(self):
        return self.op is None
    
class Tensor(Value):
    """
    A Tensor represents a multidimensional array of values in a computational graph.
    """
    

    def __init__( self, array, *, device: Optional[Device] = None, dtype=None, requires_grad=True, **kwargs):    
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                data = array.compute_cached_data()
            else:
                # fall back, copy through numpy conversion
                data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(None, (), data=data, requires_grad=requires_grad, )
        
    def __repr__(self):
        return "mi.Tensor(" + str(self.compute_cached_data()) + ")"

    def __str__(self):
        return "mi.Tensor(" + self.compute_cached_data().__str__() + ")"

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        return Tensor(self.cached_data[index])
        
    def __setitem__(self, index, value):
        self.cached_data[index] = value
        
    def _init( self, op: Optional[Operator], children: Set["Tensor"], *, num_outputs: int = 1, data: List[object] = None, requires_grad: Optional[bool] = None):
        
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(child.requires_grad for child in children)
        self.op = op
        self.cached_data = data
        self.children = children
        self.num_outputs = num_outputs
        self.requires_grad = requires_grad
        self.grad: 'Tensor'
    
    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):

        if ARRAY_API is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return ARRAY_API.array(numpy_array, device=device, dtype=dtype)
    
    @staticmethod
    def make_from_op(op: Operator, children: Tuple["Value"]):
        
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, children)
        if not LAZY_MODE:
            tensor.compute_cached_data()
        return tensor
    
    def create_detached_tensor(self, data, requires_grad=False) -> 'Tensor':
        tensor = Tensor.__new__(Tensor)
        tensor._init(None,
                     set(),
                     data=data if not isinstance(data, Tensor) else data.compute_cached_data(),
                     requires_grad=requires_grad)
        return tensor
        
        
    def detach(self) -> 'Tensor':
        return self.create_detached_tensor(self.compute_cached_data())

    @property
    def T(self) -> 'Tensor':
        return transpose(self, self.shape)
    
    def numpy(self):
        data = self.compute_cached_data()
        if ARRAY_API is numpy: return data
        return data.numpy()  # Data is of type NDArray!

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "The dtype of the given tensor (%s) is not the same as the dtype of the current tensor (%s)." % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.compute_cached_data()

    
    @property
    def shape(self):
        return self.compute_cached_data().shape

    @property
    def dtype(self):
        return self.compute_cached_data().dtype
    
    @property
    def device(self):
        data = self.compute_cached_data()
        if ARRAY_API is numpy: return cpu()
        return data.device
    
    def backward(self, out_grad: Optional['Tensor']=None) -> None:
        self.grad = out_grad if out_grad is not None else Tensor(ARRAY_API.ones(self.shape))
        
        node_to_output_grads_list: Dict[Tensor, Tensor] = {}
        node_to_output_grads_list[self] = self.grad

        def topological_sort(t) -> List['Tensor']:
            """
            Given a node in a computational graph, this function returns a list of all nodes in the graph sorted 
            in topological order.

            Args:
                self: A node in a computational graph.

            Returns:
                A list of all nodes in the graph sorted in topological order.
            """
            
            
            visited = set()
            reverse_topo_order = []

            def build_topo(node):
                visited.add(node)
                for child in node.children:
                    if child not in visited:
                        build_topo(child)
                reverse_topo_order.append(node)

            build_topo(t)
            reverse_topo_order.reverse()
            return reverse_topo_order    

        for node in topological_sort(self):
            node.grad = node_to_output_grads_list[node]
            # compute grad of current node w.r.t. output node
            # propagate grad to inputs
            if not node.is_leaf():
                for in_node, grad in zip(node.children, node.op.gradient(node.grad, node)):
                    if in_node not in node_to_output_grads_list:
                        node_to_output_grads_list[in_node] = grad
                    else:
                        node_to_output_grads_list[in_node] += grad

    
    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for addition
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for addition. Got {self.shape} and {other.shape}.")

            return EWiseAdd()(self, other)

        elif isinstance(other, (int, float)):
            return AddScalar(scalar=other)(self)

        else:
            raise ValueError(f"Unsupported operand type for +: '{type(self).__name__}' and '{type(other).__name__}'")
    
    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for subtraction
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for subtraction. Got {self.shape} and {other.shape}.")

            return EWiseAdd()(self, negate(other))

        elif isinstance(other, (int, float)):
            return AddScalar(scalar=-other)(self)

        else:
            raise ValueError(f"Unsupported operand type for -: '{type(self).__name__}' and '{type(other).__name__}'")


            
    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for multiplication
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for multiplication. Got {self.shape} and {other.shape}.")

            return EWiseMul()(self, other)

        elif isinstance(other, (int, float)):
            return MulScalar(scalar=other)(self)

        else:
            raise ValueError(f"Unsupported operand type for *: '{type(self).__name__}' and '{type(other).__name__}'")
            
    def __pow__(self, other):
        
        if isinstance(other, Tensor):
            raise NotImplementedError()        
        if isinstance(other, (int, float)):
            return PowerScalar(scalar=other)(self)
        else:
            raise ValueError(f"Unsupported operand type for ^: '{type(self).__name__}' and '{type(other).__name__}'")

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        if isinstance(other, Tensor):
            # Ensure both tensors have the same shape for addition
            if self.shape != other.shape:
                raise AssertionError(f"Tensors must be of the same shape for addition. Got {self.shape} and {other.shape}.")

            return EWiseDiv()(self, other)

        elif isinstance(other, (int, float)):
            return DivScalar(scalar=other)(self)

        else:
            raise ValueError(f"Unsupported operand type for /: '{type(self).__name__}' and '{type(other).__name__}'")

    
    def __rtruediv__(self, other): # other / self
        return self.__pow__(-1).__mul__(other)
        # other * self**-1

    def __matmul__(self, other):
        return MatMul()(self, other)
    
    
    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)
    
    def exp(self) -> 'Tensor':
        return Exp()(self)
        
    def item(self):
        return self.compute_cached_data().item()

    def argmax(self, axis=None, keepdims=None):
        return Tensor(ARRAY_API.argmax(self.compute_cached_data(), axis=axis, keepdims=keepdims))

    @staticmethod
    def accuracy(preds, yb):
       
        assert preds.shape == yb.shape
        correct_predictions = Tensor(preds.compute_cached_data() == yb.compute_cached_data()).sum()
        return correct_predictions / preds.shape[0]

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__



from numbers import Number
from typing import Optional, List
from typing import NamedTuple
import numpy

class EWiseAdd(TensorOp):    
    def compute(self, a, b): return a + b
    def gradient(self, out_grad, node): return (out_grad, out_grad)

def add(a, b): return EWiseAdd()(a, b)

# %% ../nbs/01_operators.ipynb 23
class AddScalar(TensorOp):
    def __init__(self, scalar: Union[int, float]):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (out_grad, )

def add_scalar(a: Tensor, scalar: Union[int, float]) -> Tensor:
    return AddScalar(scalar)(a)

class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.children
        return out_grad * b, out_grad * a

def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)

class MulScalar(TensorOp):
    def __init__(self, scalar: Union[int, float]):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (out_grad * self.scalar, )
    
def mul_scalar(a: Tensor, scalar: Union[int, float]) -> Tensor:
    return MulScalar(scalar)(a)

# %% ../nbs/01_operators.ipynb 32
class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.children
        return divide(out_grad, b), out_grad * negate(divide(a, power_scalar(b, 2)))


def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)

class DivScalar(TensorOp):
    def __init__(self, scalar: Union[int, float]):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        return (out_grad / self.scalar, )

def divide_scalar(a: Tensor, scalar: Union[int, float]) -> Tensor:
    return DivScalar(scalar)(a)

# %% ../nbs/01_operators.ipynb 38
class Negate(TensorOp):
    
    def compute(self, a: NDArray) -> NDArray:
        return -1 * a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor,]:
        return (negate(out_grad), )


def negate(a: Tensor) -> Tensor:
    return Negate()(a)

# %% ../nbs/01_operators.ipynb 41
class Exp(TensorOp):
    
    def compute(self, a: NDArray) -> NDArray:
        self.out = ARRAY_API.exp(a)
        return self.out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor,]:
        return (out_grad * Tensor(self.out), )

def exp(a: Tensor) -> Tensor:
    return Exp()(a)

# %% ../nbs/01_operators.ipynb 44
class ReLU(TensorOp):
    
    def compute(self, a: NDArray) -> NDArray:
        self.out = ARRAY_API.clip(a, a_min=0, a_max=None)
        return self.out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor,]:
        a = node.children[0].compute_cached_data()
        return (out_grad * Tensor(a > 0), )

def relu(a: Tensor) -> Tensor:
    return ReLU()(a)

class PowerScalar(TensorOp):

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return ARRAY_API.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ]:
        a = node.children[0]
        return (self.scalar * power_scalar(a, self.scalar - 1) * out_grad, )


def power_scalar(a: Tensor, scalar: int) -> Tensor:
    return PowerScalar(scalar)(a)

# %% ../nbs/01_operators.ipynb 53
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        if self.axes:
            a = a.swapaxes(self.axes[0], self.axes[1])
        else:
            a = a.swapaxes(-2, -1)
        return a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        return (transpose(out_grad, axes=self.axes), )

def transpose(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return ARRAY_API.reshape(a, newshape=self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, ...]:
        input_shape = node.children[0].shape
        return (reshape(out_grad, input_shape), )

def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return Reshape(shape)(a)


# %% ../nbs/01_operators.ipynb 67
class MatMul(TensorOp):    
    
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return ARRAY_API.matmul(a, b)

    
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = node.children
        out_shape, a_shape, b_shape = out_grad.shape, a.shape, b.shape
        
        # Compute the gradient with respect to a
        if len(a_shape) == len(out_shape):
            # If a and the output have the same dimensionality, we perform a matrix multiplication
            # between the output gradient and the transpose of b
            grad_wrt_a = matmul(out_grad, transpose(b))
        else:
            # If a has fewer dimensions than the output, we sum over the extra dimensions in the output
            axes_to_sum_over = tuple(range(len(out_shape) - len(a_shape)))
            grad_wrt_a = summation(matmul(out_grad, transpose(b)), axes=axes_to_sum_over)

        # Compute the gradient with respect to b
        if len(b_shape) == len(out_shape):
            # If b and the output have the same dimensionality, we perform a matrix multiplication
            # between the transpose of a and the output gradient
            grad_wrt_b = matmul(transpose(a), out_grad)
        else:
            # If b has fewer dimensions than the output, we sum over the extra dimensions in the output
            axes_to_sum_over = tuple(range(len(out_shape) - len(b_shape)))
            grad_wrt_b = summation(matmul(transpose(a), out_grad), axes=axes_to_sum_over)

        return grad_wrt_a, grad_wrt_b


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return ARRAY_API.sum(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        # out_grad is the gradient of the output of this operation
        # We need to "undo" the dimensionality reduction performed in the forward pass
        # That's why we create a new shape, replacing the dimensions specified by self.axes with 1

        # Initialize new shape to be the same as the input shape
        new_shape = list(node.children[0].shape)

        # If axes were specified, set those dimensions to 1 in the new shape
        if self.axes:
            for axis in self.axes: new_shape[axis] = 1
            
        else:
            new_shape = [1] * len(new_shape)

        # Reshape out_grad to the new shape
        reshaped_grad = reshape(out_grad, new_shape)

        # Broadcast the reshaped out_grad to match the input shape
        broadcasted_grad = broadcast_to(reshaped_grad, node.children[0].shape)

        # The gradient method needs to return a tuple, even though there's only one input
        return (broadcasted_grad, )


def summation(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return Summation(axes)(a)


# %% ../nbs/01_operators.ipynb 89
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return ARRAY_API.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        # First, we need to create a shape that matches the shape of `a` but with ones 
        # prepended to match the length of `self.shape`.
        a_shape = node.children[0].shape
        shape = [1] * (len(self.shape) - len(a_shape)) + list(a_shape)

        # Then, we identify the dimensions along which to sum in the backward pass. 
        # These are the dimensions that were expanded during the broadcast.
        sum_over = tuple([idx for idx in range(len(self.shape)) if self.shape[idx] != shape[idx]])

        # Finally, we reshape the gradient after summing over the appropriate dimensions to match `a`'s shape.
        return (reshape(summation(out_grad, sum_over), a_shape), )

def broadcast_to(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return BroadcastTo(shape)(a)

class LogSumExp(TensorOp):    
    def __init__(self, axes: Optional[tuple] = None):        
        self.axes = axes

    def compute(self, Z):
        max_z = ARRAY_API.max(Z, axis=self.axes, keepdims=True)
        self.out = ARRAY_API.squeeze(ARRAY_API.log(ARRAY_API.sum(ARRAY_API.exp(Z - max_z), axis=self.axes, keepdims=True)) + max_z)
        return self.out
    
    def gradient(self, out_grad, node):
        new_shape = list(node.children[0].shape)

        # If axes were specified, set those dimensions to 1 in the new shape
        if self.axes:
            for axis in self.axes: new_shape[axis] = 1
        else:
            new_shape = [1] * len(new_shape)
        
        if self.axes:
            reshaped_grad = reshape(out_grad, new_shape)
            reshaped_out = reshape(node, new_shape)

            # Broadcast the reshaped out_grad to match the input shape
            broadcasted_grad = broadcast_to(reshaped_grad, node.children[0].shape)
            broadcasted_out = broadcast_to(reshaped_out, node.children[0].shape)
            return (exp(node.children[0] - broadcasted_out) * broadcasted_grad, )
        return (exp(node.children[0] - self.out) * out_grad, )
    
def logsumexp(a, axes=None): 
    return LogSumExp(axes=axes)(a)