# Chapter 3 - Introduction to Tensorflow, PyTorch, JAX, and Keras

## A brief history of deep learning frameworks

- Three key features of deep learning frameworks:

1. A way to compute gradients for arbitrary differentiable functions (automatic differentiation).
2. A way to run tensor computations on CPUs and GPUs (and possibly even on other specialized deep learning hardware).
3. A way to distribute computation across multiple devices or multiple computers, such as multiple GPUs on one computer, or even multiple GPUs across multiple separate computers.

- XLA - A high-performance compiler develop to enable Tensorflow to run on TPUs.

- Tensorflow: Released in 2015 by google
- PyTorch: Released in 2016 by Meta.
- JAX: An alternative way to use autodifferentiation with XLA, Google.

## How these frameworks relate to each other

- Low-level Frameworks: Tensor manipulation (tensors, tensor operations (addition, relu, matmul), backpropagation)
    1. Tensorflow
    2. PyTorch
    3. JAX

- High-level Frameworks: High-level deep learning concepts (layers, loss-functions, optimizers, metrics, training lloop that performs mini-batch Stochastic Gradient Descent)
    1. Keras

## Introduction to Tensorflow

### Tensors and variables in Tensorflow

- Constant Tensors: Tensors need to be created with some initial value, so common ways to create tensors are via `tf.ones` (equivalent to np.ones) and `tf.zeros` (equivalent to np.zeros). 
    - You can also create a tensor from Python or NumPy values using tf.constant.

    ```python
    >>> import tensorflow as tf
    >>> # Equivalent to np.ones(shape=(2, 1))
    >>> tf.ones(shape=(2, 1))
    tf.Tensor([[1.], [1.]], shape=(2, 1), dtype=float32)
    >>> # Equivalent to np.zeros(shape=(2, 1))
    >>> tf.zeros(shape=(2, 1))
    tf.Tensor([[0.], [0.]], shape=(2, 1), dtype=float32)
    >>> # Equivalent to np.array([1, 2, 3], dtype="float32")
    >>> tf.constant([1, 2, 3], dtype="float32")
    tf.Tensor([1., 2., 3.], shape=(3,), dtype=float32)
    ```
- Random Tensors: You can also create tensors filled with random values via one of the methods of the `tf.random` submodule (equivalent to the np.random submodule).

    ```python
    >>> # Tensor of random values drawn from a normal distribution with mean 0 and standard deviation 1. Equivalent to np.random.normal(size=(3, 1), loc=0., scale=1.).
    >>> x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
    >>> print(x)
    tf.Tensor(
    [[-0.14208166]
    [-0.95319825]
    [ 1.1096532 ]], shape=(3, 1), dtype=float32)
    >>> # Tensor of random values drawn from a uniform distribution between 0 and 1. Equivalent to np.random.uniform(size=(3, 1), low=0.,high=1.).
    >>> x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
    >>> print(x)
    tf.Tensor(
    [[0.33779848]
    [0.06692922]
    [0.7749394 ]], shape=(3, 1), dtype=float32)
    ```

### Tensor assignment and the variable class
- TensorFlow tensors are constants and therefore not assignable.
    - `tf.Variable`: A class meant to manage modifiable state in TensorFlow.
        - To create a variable you need to provide some initial value such as a random tensor.
        ```python
            >>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
            >>> print(v)
            array([[-0.75133973],
                [-0.4872893 ],
                [ 1.6626885 ]], dtype=float32)>
        ```
        - The state of the variable can be modified with the `assign` method.

        ```python
        >>> v.assign(tf.ones((3, 1)))
            array([[1.],
                [1.],
                [1.]], dtype=float32)>
        ```
        - Assignment also works for subset of the coefficients.
        ```python
        >>> v[0, 0].assign(3.)
            array([[3.],
                [1.],
                [1.]], dtype=float32)>
        ```
        - `assign_add` and `assign_sub` are efficient equivalents of `+=` and `-=`.
        ```python
        >>> v.assign_add(tf.ones((3, 1)))
            array([[2.],
                [2.],
                [2.]], dtype=float32)>
        ```

### Tensor operations: Doing math in TensorFlow
```python
    a = tf.ones((2, 2))
    # Takes the square, same as np.square
    b = tf.square(a)
    # Takes the square root, same as np.sqrt
    c = tf.sqrt(a)
    # Adds two tensors (element-wise)
    d = b + c
    # Takes the product of two tensors (see chapter 2), same as np.matmul
    e = tf.matmul(a, b)
    # Concatenates a and b along axis 0, same as np.concatenate
    f = tf.concat((a, b), axis=0)
```
- An equivalent to the Dense layer in keras for Tensorflow:
    ```python
        def dense(inputs, W, b):
            return tf.nn.relu(tf.matmul(inputs, W) + b)
    ```

## Gradients in TensorFlow: A second look at the GradientTape API

```python
    input_var = tf.Variable(initial_value=3.0)
    with tf.GradientTape() as tape:
        result = tf.square(input_var)
    gradient = tape.gradient(result, input_var)
```
- Commonly used to retrieve the gradients of the loss of a model with respect to its weights: `gradients = tape.gradient(loss, weights)`

- Manually call `tape.watch()` to track constant tensors that are not variables.
```python
    input_const = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(input_const)
        result = tf.square(input_const)
    gradient = tape.gradient(result, input_const)
```
**NOTE**: gradient tape is a powerful utility, even capable of computing `second-order gradients` — that is, the gradient of a gradient
```python
    time = tf.Variable(0.0)
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            position = 4.9 * time**2
        speed = inner_tape.gradient(position, time)
    # We use the outer tape to compute the gradient of the gradient from the inner tape. Naturally, the answer is 4.9 * 2 = 9.8.
    acceleration = outer_tape.gradient(speed, time)
```