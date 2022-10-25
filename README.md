<div align="center">
  <h1><code>NumPy Tasks from Beginner to Advanced Level</code></h1>

  <p>
    <strong>A NumPy Task by 
    <a href="https://www.guvi.in/">GUVI - Greek Networks</a></strong>
  </p>

  <strong>A <a href="https://numpy.org/">NumPy</a> project Q/A</strong>

  <h3>
    <a href="https://numpy.org/doc/stable/user/">Guide by Numpy.org</a>
    <span>
  </h3>
</div>

# Numpy

#### 1. Import the numpy package under the name `np` (★☆☆) 
(**hint**: import … as …)
```python
#import library 
import numpy as np
```
#### 2. Print the numpy version and the configuration (★☆☆) 
(**hint**: np.\_\_version\_\_, np.show\_config)
```python
import numpy as np
#To Show version of NumPy
print(np.__version__)
```
Output:
```
1.21.5
```
<div align="left">
  <h6><code>Or</code></h6>
  
```python
import numpy as np
#to Show Configuration of NumPy in python
np.show_config()
#Or
np.__config__.show()
```
Output: 
```
#will be about

blas_mkl_info:
blas_opt_info:
lapack_mkl_info:
lapack_opt_info:
```
#### 3. Create a null vector of size 10 (★☆☆) 
(**hint**: np.zeros)
```python
import numpy as np
# create a numpy null vector os size 10
def null_array(n):
    x=np.zeros(n)
    return x
print(null_array(10))
```
#Or
```
import numpy as np
X = np.zeros(10)
print(X)
```
Output:
```
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

#### 4.  How to find the memory size of any array (★☆☆) 
(**hint**: size, itemsize)
```python
#Example:
import numpy as np
x = np.array([100,20,34])     #create a numpy 1d-array
print(x.size)                 #Size of the array
print(x.itemsize)             #Memory size of one array element in bytes
```
Output:
```
3
4
```

#### 5.How to get the documentation of the numpy add function from the command line? (★☆☆) 
(**hint**: np.info)
```python
import numpy as np
np.info(np.add)
```
#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
(**hint**: array\[4\])
```python
import numpy as np
X= np.zeros(10)
X[4]= 1
print (X)
```
Output:
```
[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
```

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 
(**hint**: np.arange)
```python
import numpy as np
X=np.arange(10,50)
print(X)
```
Output:
```
[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
```

#### 8.  Reverse a vector (first element becomes last) (★☆☆) 
(**hint**: array\[::-1\])
```python
import numpy as np
array = np.arange(10,50)
array = array[::-1]
print(array)
```
Output:
```
[49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26
 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10]
```

#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
(**hint**: reshape)
```python
import numpy as np
X = np.arange(9).reshape(3,3)
print (X)
```
Output:
```
[[0 1 2]
 [3 4 5]
 [6 7 8]]
 ```
 
#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
(**hint**: np.nonzero)
```python
import numpy as np
n_z = np.nonzero([1,2,0,0,4,0])
print(n_z)
```
Output:
```
(array([0, 1, 4], dtype=int64),)
```

#### 11. Create a 3x3 identity matrix (★☆☆) 
(**hint**: np.eye)
```python
import numpy as np
X = np.eye(3)
print(X)
```
Output:
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

#### 12. Create a 3x3x3 array with random values (★☆☆) 
(**hint**: np.random.random)
```python
import numpy as np
X = np.random.random((3,3,3))
print(X)
```
Output:
```
[[[0.49900755 0.28634814 0.07203155]
  [0.68515457 0.4245794  0.9286755 ]
  [0.27260579 0.40500071 0.37400368]]

 [[0.33080571 0.48782597 0.10603305]
  [0.48308991 0.12279789 0.34028278]
  [0.07807634 0.60300338 0.11959647]]

 [[0.83295228 0.89194862 0.10970957]
  [0.65073572 0.19738613 0.76277278]
  [0.5032257  0.62214717 0.96179147]]]
 ```

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
(**hint**: min, max)
```python
import numpy as np
X = np.random.random((10,10)) #to print 10*10 random matrix
print(X)
Xmin, Xmax = X.min(), X.max() #to print Minimum and Maximum values in 10*10 random Matrix
print(Xmin, Xmax)
```
Output:
```
[[0.91600365 0.43388054 0.71710045 0.92099685 0.60942653 0.22180512
  0.28675219 0.98204934 0.68376251 0.22466073]
 [0.85372035 0.51053317 0.0414746  0.11099836 0.97099723 0.15735309
  0.64174022 0.13736961 0.07286516 0.7447888 ]
 [0.32479725 0.84624883 0.46314502 0.07544871 0.98570801 0.32425058
  0.85489316 0.56927101 0.29662688 0.85275016]
 [0.56841332 0.18406031 0.05567367 0.14478287 0.83482955 0.00564811
  0.03299423 0.58081476 0.7775403  0.41803937]
 [0.43649209 0.65684646 0.22514752 0.69408093 0.87164318 0.00445458
  0.13859695 0.23829343 0.74470312 0.41937707]
 [0.85511553 0.32132907 0.06347097 0.69730243 0.9534301  0.4226164
  0.42844068 0.27996545 0.88012548 0.92392705]
 [0.12645999 0.25502506 0.571237   0.5984796  0.89611264 0.68004171
  0.15307523 0.25241456 0.39253602 0.88220187]
 [0.30822073 0.27348611 0.46475688 0.8911376  0.49759933 0.95730246
  0.92658798 0.29119787 0.20752352 0.39373945]
 [0.02417271 0.62831929 0.77902002 0.43110769 0.13500418 0.50095792
  0.15833488 0.24539258 0.22124239 0.00677951]
 [0.22731453 0.15096889 0.24403781 0.82689975 0.04262165 0.62918284
  0.6267458  0.38301252 0.81355988 0.3499431 ]]
0.004454584306314069 0.9857080060582043                              
````

#### 14. Create a random vector of size 30 and find the mean value (★☆☆) 
(**hint**: mean)
```python
import numpy as np
X = np.random.random(10)
print(X)
mean = X.mean()
print(mean)
```
Output:
```
[0.82161788 0.85385135 0.33944699 0.52646573 0.59985237 0.94029297
 0.87821587 0.75339441 0.17994566 0.31976549]
0.6212848734289441
```

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
(**hint**: array\[1:-1, 1:-1\])
```python
import numpy as np
array = np.ones((10,10))
array[1:-1, 1:-1]=0
```

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 
(**hint**: np.pad)

```python
have to do
```

#### 17. What is the result of the following expression? (★☆☆) 
(**hint**: NaN = not a number, inf = infinity)

```
#Even python code
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```
```python
import numpy as np
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
```
Output:
```
nan
False
False
nan
False
```

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
(**hint**: np.diag)
```python
import numpy as np
X = np.diag(1+np.arange(4), k = -1)
print (X)
```
Output:
```
[[0 0 0 0 0]
 [1 0 0 0 0]
 [0 2 0 0 0]
 [0 0 3 0 0]
 [0 0 0 4 0]]
```

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 
(**hint**: array\[::2\])
```python
import numpy as np
array = np.zeros ((8,8), dtype=int)
array[1::2, ::2]= 1
array[::2, 1::2] = 1
print (array)
```
Output:
```
[[0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]]
```

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 
(**hint**: np.unravel_index)
```python
import numpy as np
print(np.unravel_index(100, (6,7,8))) 
```
Output:
```
(1, 5, 4)
```

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 
(**hint**: np.tile)
```python
import numpy as np
array= np.array([[0,1], [1,0]])
X = np.tile(array,(4,4))
print (X)
```
Output:
```
[[0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]]
```

#### 22. Normalize a 5x5 random matrix (★☆☆) 
(**hint**: (x - min) / (max - min))
```python
import numpy as np
X = np.random.random((5,5))
Xmax, Xmin = X.max(), X.min()
X= (X-Xmin)/(Xmax-Xmin)
print(X)
```
Output:
```
[[0.         0.49565479 0.34481009 0.67883924 0.92754924]
 [0.6401139  0.22507495 0.94955545 0.45722861 0.86650414]
 [0.03730368 0.5169144  0.13886795 0.79054127 0.8018405 ]
 [0.62536276 0.02830271 0.50317556 0.7566859  0.59502691]
 [0.14215049 0.95454916 0.54981254 1.         0.04512418]]
```

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) 
(**hint**: np.dtype)
```python
#have to do
```

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 
(**hint**: np.dot | @)
```python
import numpy as np
X = np.dot(np.ones((5,3)), np.ones((3,2)))
print(X)
```
Output:
```
[[3. 3.]
 [3. 3.]
 [3. 3.]
 [3. 3.]
 [3. 3.]]
```

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) 
(**hint**: >, <=)
```python
have to do
```

#### 26. What is the output of the following script? (★☆☆) 
(**hint**: np.sum)

```python
import numpy as np
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
Output:
```
9
10
```

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
```python
import numpy as np
#Let Z = 5 to find which of these expressions are legal.
Z = 5
print(Z**Z)
print(2 << Z >> 2)
print(Z <- Z)
print(1j*Z)
print(Z/1/1)
print(Z<Z>Z)
```
Output:
```
3125
16
False
5j
5.0
False
```

#### 28. What are the result of the following expressions?

```python
import numpy as np
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
  
#### 29. How to round away from zero a float array ? (★☆☆) 
(**hint**: np.uniform, np.copysign, np.ceil, np.abs)
```python
import numpy as np
def round_array(x,y):
    return np.round(x,y)                  #Defining round_array function
test = np.array([32.11, 51.5, 0.112])     #Setup the Data
print(round_array(test,0))                #Printing rounded off array
```
Output:
```
[32. 52.  0.]
```

  #### 30. How to find common values between two arrays? (★☆☆) 
(**hint**: np.intersect1d)
```python
import numpy as np
ar1 = np.array([0, 1, 2, 3, 4, 10, 7, 6, 9])
ar2 = [1, 3, 4, 0, 5, 9, 22, 29]
print(np.intersect1d(ar1, ar2))                 # Common values between two arrays
```
Output:
```
[0 1 3 4 9]
```

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) 
(**hint**: np.seterr, np.errstate)
```python
import numpy as np
data = np.random.random(1000).reshape(10, 10,10) * np.nan
np.seterr(all="ignore")
np.nanmedian(data, axis=[1, 2])
print(data)
```
Or
```python
import numpy as np
data = np.random.random(1000).reshape(10, 10,10) * np.nan
np.errstate(all="ignore")
np.nanmedian(data, axis=[1, 2])
print(data)
```
Output:
```
[[[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]

 [[nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]
  [nan nan nan nan nan nan nan nan nan nan]]]
  ```



#### 32. Is the following expressions true? (★☆☆) 
(**hint**: imaginary number)

```python
np.sqrt(-1) == np.emath.sqrt(-1)
```



#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 
(**hint**: np.datetime64, np.timedelta64)



#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 
(**hint**: np.arange(dtype=datetime64\['D'\]))



#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 
(**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))



#### 36. Extract the integer part of a random array using 5 different methods (★★☆) 
(**hint**: %, np.floor, np.ceil, astype, np.trunc)



#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
(**hint**: np.arange)



#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 
(**hint**: np.fromiter)



#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
(**hint**: np.linspace)



#### 40. Create a random vector of size 10 and sort it (★★☆) 
(**hint**: sort)



#### 41. How to sum a small array faster than np.sum? (★★☆) 
(**hint**: np.add.reduce)



#### 42. Consider two random array A and B, check if they are equal (★★☆) 
(**hint**: np.allclose, np.array\_equal)



#### 43. Make an array immutable (read-only) (★★☆) 
(**hint**: flags.writeable)



#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 
(**hint**: np.sqrt, np.arctan2)



#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
(**hint**: argmax)



#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 
(**hint**: np.meshgrid)



####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 
(**hint**: np.subtract.outer)



#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 
(**hint**: np.iinfo, np.finfo, eps)



#### 49. How to print all the values of an array? (★★☆) 
(**hint**: np.set\_printoptions)



#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 
(**hint**: argmin)



#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 
(**hint**: dtype)



#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 
(**hint**: np.atleast\_2d, T, np.sqrt)



#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 
(**hint**: astype(copy=False))



#### 54. How to read the following file? (★★☆) 
(**hint**: np.genfromtxt)

```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```



#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 
(**hint**: np.ndenumerate, np.ndindex)



#### 56. Generate a generic 2D Gaussian-like array (★★☆) 
(**hint**: np.meshgrid, np.exp)



#### 57. How to randomly place p elements in a 2D array? (★★☆) 
(**hint**: np.put, np.random.choice)



#### 58. Subtract the mean of each row of a matrix (★★☆) 
(**hint**: mean(axis=,keepdims=))



#### 59. How to sort an array by the nth column? (★★☆) 
(**hint**: argsort)



#### 60. How to tell if a given 2D array has null columns? (★★☆) 
(**hint**: any, ~)



#### 61. Find the nearest value from a given value in an array (★★☆) 
(**hint**: np.abs, argmin, flat)



#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 
(**hint**: np.nditer)



#### 63. Create an array class that has a name attribute (★★☆) 
(**hint**: class method)



#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 
(**hint**: np.bincount | np.add.at)



#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 
(**hint**: np.bincount)



#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 
(**hint**: np.unique)



#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 
(**hint**: sum(axis=(-2,-1)))



#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 
(**hint**: np.bincount)



#### 69. How to get the diagonal of a dot product? (★★★) 
(**hint**: np.diag)



#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
(**hint**: array\[::4\])



#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
(**hint**: array\[:, :, None\])



#### 72. How to swap two rows of an array? (★★★) 
(**hint**: array\[\[\]\] = array\[\[\]\])



#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 
(**hint**: repeat, np.roll, np.sort, view, np.unique)



#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 
(**hint**: np.repeat)



#### 75. How to compute averages using a sliding window over an array? (★★★) 
(**hint**: np.cumsum)



#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 
(**hint**: from numpy.lib import stride_tricks)



#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 
(**hint**: np.logical_not, np.negative)



#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)



#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)



#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 
(**hint**: minimum, maximum)



#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 
(**hint**: stride\_tricks.as\_strided)



#### 82. Compute a matrix rank (★★★) 
(**hint**: np.linalg.svd) (suggestion: np.linalg.svd)



#### 83. How to find the most frequent value in an array? 
(**hint**: np.bincount, argmax)



#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 
(**hint**: stride\_tricks.as\_strided)



#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 
(**hint**: class method)



#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 
(**hint**: np.tensordot)



#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) 
(**hint**: np.add.reduceat)



#### 88. How to implement the Game of Life using numpy arrays? (★★★)



#### 89. How to get the n largest values of an array (★★★) 
(**hint**: np.argsort | np.argpartition)



#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) 
(**hint**: np.indices)



#### 91. How to create a record array from a regular array? (★★★) 
(**hint**: np.core.records.fromarrays)



#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) 
(**hint**: np.power, \*, np.einsum)



#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) 
(**hint**: np.where)



#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)



#### 95. Convert a vector of ints into a matrix binary representation (★★★) 
(**hint**: np.unpackbits)



#### 96. Given a two dimensional array, how to extract unique rows? (★★★) 
(**hint**: np.ascontiguousarray)



#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) 
(**hint**: np.einsum)



#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? 
(**hint**: np.cumsum, np.interp)



#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) 
(**hint**: np.logical\_and.reduce, np.mod)



#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) 
(**hint**: np.percentile)



## Installation

The Wasmtime CLI can be installed on Linux and macOS with a small install
script:

```sh
curl https://wasmtime.dev/install.sh -sSf | bash
```

Windows or otherwise interested users can download installers and
binaries directly from the [GitHub
Releases](https://github.com/bytecodealliance/wasmtime/releases) page.

## Example

If you've got the [Rust compiler
installed](https://www.rust-lang.org/tools/install) then you can take some Rust
source code:

```rust
fn main() {
    println!("Hello, world!");
}
```

and compile/run it with:

```sh
$ rustup target add wasm32-wasi
$ rustc hello.rs --target wasm32-wasi
$ wasmtime hello.wasm
Hello, world!
```

## Features

* **Fast**. Wasmtime is built on the optimizing [Cranelift] code generator to
  quickly generate high-quality machine code either at runtime or
  ahead-of-time. Wasmtime is optimized for efficient instantiation, low-overhead
  calls between the embedder and wasm, and scalability of concurrent instances.

* **[Secure]**. Wasmtime's development is strongly focused on correctness and
  security. Building on top of Rust's runtime safety guarantees, each Wasmtime
  feature goes through careful review and consideration via an [RFC
  process]. Once features are designed and implemented, they undergo 24/7
  fuzzing donated by [Google's OSS Fuzz]. As features stabilize they become part
  of a [release][release policy], and when things go wrong we have a
  well-defined [security policy] in place to quickly mitigate and patch any
  issues. We follow best practices for defense-in-depth and integrate
  protections and mitigations for issues like Spectre. Finally, we're working to
  push the state-of-the-art by collaborating with academic researchers to
  formally verify critical parts of Wasmtime and Cranelift.

* **[Configurable]**. Wasmtime uses sensible defaults, but can also be
  configured to provide more fine-grained control over things like CPU and
  memory consumption. Whether you want to run Wasmtime in a tiny environment or
  on massive servers with many concurrent instances, we've got you covered.

* **[WASI]**. Wasmtime supports a rich set of APIs for interacting with the host
  environment through the [WASI standard](https://wasi.dev).

* **[Standards Compliant]**. Wasmtime passes the [official WebAssembly test
  suite](https://github.com/WebAssembly/testsuite), implements the [official C
  API of wasm](https://github.com/WebAssembly/wasm-c-api), and implements
  [future proposals to WebAssembly](https://github.com/WebAssembly/proposals) as
  well. Wasmtime developers are intimately engaged with the WebAssembly
  standards process all along the way too.

[Wasmtime]: https://github.com/bytecodealliance/wasmtime
[Cranelift]: https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/README.md
[Google's OSS Fuzz]: https://google.github.io/oss-fuzz/
[security policy]: https://bytecodealliance.org/security
[RFC process]: https://github.com/bytecodealliance/rfcs
[release policy]: https://docs.wasmtime.dev/stability-release.html
[Secure]: https://docs.wasmtime.dev/security.html
[Configurable]: https://docs.rs/wasmtime/latest/wasmtime/struct.Config.html
[WASI]: https://docs.rs/wasmtime-wasi/latest/wasmtime_wasi/
[Standards Compliant]: https://docs.wasmtime.dev/stability-wasm-proposals-support.html

## Language Support

You can use Wasmtime from a variety of different languages through embeddings of
the implementation:

* **[Rust]** - the [`wasmtime` crate]
* **[C]** - the [`wasm.h`, `wasi.h`, and `wasmtime.h` headers][c-headers], [CMake](crates/c-api/CMakeLists.txt) or [`wasmtime` Conan package]
* **C++** - the [`wasmtime-cpp` repository][wasmtime-cpp] or use [`wasmtime-cpp` Conan package]
* **[Python]** - the [`wasmtime` PyPI package]
* **[.NET]** - the [`Wasmtime` NuGet package]
* **[Go]** - the [`wasmtime-go` repository]

[Rust]: https://bytecodealliance.github.io/wasmtime/lang-rust.html
[C]: https://bytecodealliance.github.io/wasmtime/examples-c-embed.html
[`wasmtime` crate]: https://crates.io/crates/wasmtime
[c-headers]: https://bytecodealliance.github.io/wasmtime/c-api/
[Python]: https://bytecodealliance.github.io/wasmtime/lang-python.html
[`wasmtime` PyPI package]: https://pypi.org/project/wasmtime/
[.NET]: https://bytecodealliance.github.io/wasmtime/lang-dotnet.html
[`Wasmtime` NuGet package]: https://www.nuget.org/packages/Wasmtime
[Go]: https://bytecodealliance.github.io/wasmtime/lang-go.html
[`wasmtime-go` repository]: https://pkg.go.dev/github.com/bytecodealliance/wasmtime-go
[wasmtime-cpp]: https://github.com/bytecodealliance/wasmtime-cpp
[`wasmtime` Conan package]: https://conan.io/center/wasmtime
[`wasmtime-cpp` Conan package]: https://conan.io/center/wasmtime-cpp

## Documentation

[📚 Read the Wasmtime guide here! 📚][guide]

The [wasmtime guide][guide] is the best starting point to learn about what
Wasmtime can do for you or help answer your questions about Wasmtime. If you're
curious in contributing to Wasmtime, [it can also help you do
that][contributing]!

[contributing]: https://bytecodealliance.github.io/wasmtime/contributing.html
[guide]: https://bytecodealliance.github.io/wasmtime

---

It's Wasmtime.

<h3>
    <a href="https://www.linkedin.com/in/venkatesh-kadali/">Chat on LinkedIn</a>
    <span> | </span>
  </h3>
