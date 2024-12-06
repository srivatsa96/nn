import numpy as np

'''
A tensor object that support upto 2 dimension object that 
support major operations like addition, substraction, element wise operations (multiplication, non lin function)
and matrix multiplication.

Limitation:
1. Currently supports upto 2d matrices only.
2. Broacasting limited to addition only
3. Broadcasting of element wise multiplication yet not tested

'''
class Tensor:

    def __init__(self, data, _children = (), _op = ''):
        self.data = np.array(data)  # Store as a numpy 2D array
        self.grad = np.zeros_like(self.data)  # Gradient initialized to zero
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # Operation that produced this node
    
    def __add__(self, other):
        ## Convert Scalars to 1 x 1 tensor
        if isinstance(other, (int, float)):
            other = Tensor(np.asarray([other],dtype=np.float32).reshape(1,1))

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data + other.data, (self, other), '+')

        ## In case of Addition of two matrix, the incoming gradient is 
        ## is equally split between operends.
        def _backward():
            
            if self.data.shape == out.grad.shape:
                self.grad += out.grad
            ## Handle Broadcasted addition. Gradient is summed along the first K dimensions of output along which the smaller tensor was broadcasted
            else:
                
                self.grad += np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(self.grad.shape))))
                
            
            if other.data.shape == out.grad.shape:
                other.grad += out.grad
            ## Handle Broadcasted addition. Gradient is summed along the first K dimensions of output along which the smaller tensor was broadcasted
            else:
                other.grad += np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(other.grad.shape))))

        out._backward = _backward

        return out
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.asarray([other], dtype=np.float32).reshape(1, 1))
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # Create a function to generate axes for summing gradients
            def get_broadcast_axes(shape1, shape2):
                axes = []
                # Reverse iterate through both shapes
                for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
                    if dim1 != dim2:
                        axes.append(len(shape1) - len(axes) - 1)  # Add the axis index
                return tuple(reversed(axes))  # Reverse to original order

            # Handle gradients for self
            if self.data.shape == out.data.shape:
                self.grad += other.data * out.grad
            else:
                axes_self = get_broadcast_axes(out.data.shape, self.data.shape)
                self.grad += np.sum(other.data * out.grad, axis=axes_self, keepdims=True)

            # Handle gradients for other
            if other.data.shape == out.data.shape:
                other.grad += self.data * out.grad
            else:
                axes_other = get_broadcast_axes(out.data.shape, other.data.shape)
                other.grad += np.sum(self.data * out.grad, axis=axes_other, keepdims=True)

        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        # if not isinstance(other, Tensor):
        #     raise ValueError(f"matmul only supports Tensor instances. Got {type(other)}")
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(f"Tensor multiplication requires compatible dimensions: {self.data.shape[1]} (columns of A) != {other.data.shape[0]} (rows of B)")

        out = Tensor(np.dot(self.data, other.data), (self, other), '@')
        def _backward():
            self.grad += np.dot(out.grad, other.data.T)  # Chain rule for matrix multiplication -> Local Gradient X Incoming Gradient
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(np.power(self.data, other), (self,), f'**{other}')

        def _backward():
            self.grad += (other * np.power(self.data, other - 1)) * out.grad # -> Local Gradient X Incoming Gradient
        out._backward = _backward

        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad  # Element-wise gradient
            
        out._backward = _backward

        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), "Tanh")

        def _backward():
            self.grad += (1-np.tanh(self.data)**2)*out.grad
        
        out._backward = _backward

        return out
    
    def sigmoid(self):

        def _compute_sigmoid(x):
            return 1/(1 + np.exp(-1*x))
        
        out = Tensor(_compute_sigmoid(self.data),(self,), "sigmoid")

        def _backward():
            self.grad = out.data * (1 - out.data) * out.grad
        
        out._backward = _backward

        return out
    
    def item(self):
        if self.data.shape == (1,):
            return self.data[0]
        elif self.data.shape == (1,1):
            return self.data[0][0]
        elif self.data.shape == ():
            return self.data
        else:
            raise ValueError("Only Single Item Matrix supports item")
        
    ## Horizontal Stack Numpy Array
    def hstack(self,other):
         return Tensor(np.hstack((self.data,other)))

    ## Vertical Stack Numpy Array
    def vstack(self,other):
        return Tensor(np.vstack((self.data,other)))
    
    '''Where magic happens'''
    def backward(self):
        if not (self.data.shape == (1,) or self.data.shape == (1,1) or self.data.shape == ()):
            raise ValueError('Backward can be called on scalars only')

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones(self.data.shape)
        # for v in reversed(topo):
        #     if(v._op.find('weight')!=-1 or v._op.find('bias')!=-1):
        #         print('in+back', v._op, v.grad)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * Tensor(np.full(self.data.shape, -1))

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    @property
    def shape(self):
        return self.data.shape

    
