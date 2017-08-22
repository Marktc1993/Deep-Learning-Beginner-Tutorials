
# Here I am going to explore the reasons why rank 1 arrays should be avoided when designing neural networks in Python 
#### from Deep Learning by Andrew Ng

#### Transcribed by Mark Conrad


### Introduction:
Lets start by importing the numpy library and creating an array named a.


```python
import numpy as np
a = np.random.randn(5) # here we are defining a vector with five random Guassian values
print(a)
```

    [-0.8595348  -0.22223392  0.10085827 -0.43350044 -0.03448499]



```python
a.shape 
```




    (5,)



Here we see that we have defined a rank 1 array in numpy which is neither a row or  column vector. Let's perform the tranpose operation and see what happens.


```python
print(a.T)
```

    [-0.8595348  -0.22223392  0.10085827 -0.43350044 -0.03448499]


We see that this the exact same array. Let's try printing the dot product of the tranposed array and the original.


```python
print(np.dot(a, a.T))
```

    0.98747223211


Here we get back a scalar, eek! Instead of designating a to be a rank 1 array, let's explicitly make row and column vectors


```python
a = np.random.randn(5,1) # this creates a 5x1 column vector
print(a)
```

    [[-0.1421161 ]
     [-1.02092054]
     [ 0.87825347]
     [-0.32616055]
     [-1.90971294]]


Let's see what a transpose looks like:


```python
print(a.T)
```

    [[-0.1421161  -1.02092054  0.87825347 -0.32616055 -1.90971294]]


Here we see that we have converted our column vector into a row vector, or more concretely a 1x5 matrix.

Now if we perform the dot product between these two vectors we should get the outer product of a vector or in other words the matrix.


```python
print(np.dot(a,a.T))
```

    [[ 0.02019699  0.14508924 -0.12481396  0.04635266  0.27140095]
     [ 0.14508924  1.04227874 -0.89662701  0.33298401  1.94966516]
     [-0.12481396 -0.89662701  0.77132916 -0.28645164 -1.67721202]
     [ 0.04635266  0.33298401 -0.28645164  0.10638071  0.62287303]
     [ 0.27140095  1.94966516 -1.67721202  0.62287303  3.6470035 ]]


Suffice it to say, we do not want to use rank 1 arrays, they can lead to really weird bugs and are generally unnecessary in neural networks. To avoid issues with Python and broadcasting, we can assert or use reshape to be sure that the arrays act more consistently to that of row and column vectors. 


```python
assert(a.shape == (5,1))# this double checks that a is to be a column vector, and is not computationally expensive.
print(a)
```

    [[-0.1421161 ]
     [-1.02092054]
     [ 0.87825347]
     [-0.32616055]
     [-1.90971294]]



```python
# ALTERNATIVELY
a = a.reshape(5,1)
print(a)
```

    [[-0.1421161 ]
     [-1.02092054]
     [ 0.87825347]
     [-0.32616055]
     [-1.90971294]]


### In conclusion, 
Always use row and column vectors, and use assertions to double check that you are in fact not using rank 1 arrays!


```python

```
