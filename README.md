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
```python
import numpy as np
np.sqrt(-1) == np.emath.sqrt(-1)
```
Output:
```
False
```
#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 
(**hint**: np.datetime64, np.timedelta64)
```python
import numpy as np
  
today = np.datetime64('today', 'D')           # for today
print("Today: ", today)

yesterday = np.datetime64('today', 'D')       # for yesterday
- np.timedelta64(1, 'D')
print("Yestraday: ", yesterday)

tomorrow = np.datetime64('today', 'D')        # for tomorrow
+ np.timedelta64(1, 'D') 
print("Tomorrow: ", tomorrow)
```
Output:
```
Today:  2022-10-25
Yestraday:  2022-10-25
Tomorrow:  2022-10-25
```

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 
(**hint**: np.arange(dtype=datetime64\['D'\]))
```python
import numpy as np
print("Month: July, Year: 2016")
print(np.arange('2016-07', '2016-08', dtype='datetime64[D]'))
```
Output:
```
Month: July, Year: 2016
['2016-07-01' '2016-07-02' '2016-07-03' '2016-07-04' '2016-07-05'
 '2016-07-06' '2016-07-07' '2016-07-08' '2016-07-09' '2016-07-10'
 '2016-07-11' '2016-07-12' '2016-07-13' '2016-07-14' '2016-07-15'
 '2016-07-16' '2016-07-17' '2016-07-18' '2016-07-19' '2016-07-20'
 '2016-07-21' '2016-07-22' '2016-07-23' '2016-07-24' '2016-07-25'
 '2016-07-26' '2016-07-27' '2016-07-28' '2016-07-29' '2016-07-30'
 '2016-07-31']
```

#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 
(**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))
```
have to do
```

#### 36. Extract the integer part of a random array using 5 different methods (★★☆) 
(**hint**: %, np.floor, np.ceil, astype, np.trunc)
```python
import numpy as np
  
Z = np.random.uniform(0,10,10)
print(help(np.random.uniform))

print('Z:\n',Z)
print (Z-Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))
```
Output:
```
[6. 1. 7. 1. 0. 2. 0. 0. 0. 4.]
[6. 1. 7. 1. 0. 2. 0. 0. 0. 4.]
[6. 1. 7. 1. 0. 2. 0. 0. 0. 4.]
[6 1 7 1 0 2 0 0 0 4]
[6. 1. 7. 1. 0. 2. 0. 0. 0. 4.]
```

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
(**hint**: np.arange)
```python
import numpy as np

Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)
```
Output:
```
[[0. 1. 2. 3. 4.]
 [0. 1. 2. 3. 4.]
 [0. 1. 2. 3. 4.]
 [0. 1. 2. 3. 4.]
 [0. 1. 2. 3. 4.]]
```

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 
(**hint**: np.fromiter)
```python
import numpy as np
def generate(): 
    for x in range(10):
        yield x
        
Z = np.fromiter(generate(), dtype=float, count=-1)
print (Z)
```
Output:
```
[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
```

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
(**hint**: np.linspace)
```python
import numpy as np
X = np.linspace(0,1,12,endpoint=True)[1:-1]
print(X)
```
Output:
```
[0.09090909 0.18181818 0.27272727 0.36363636 0.45454545 0.54545455
 0.63636364 0.72727273 0.81818182 0.90909091]
```

#### 40. Create a random vector of size 10 and sort it (★★☆) 
(**hint**: sort)
```python
import numpy as np
X = np.random.random(10)
X.sort()
print(X)
```
Output:
```
[0.07300256 0.1144625  0.2303744  0.3376685  0.38007836 0.46996726
 0.48838967 0.66329321 0.72652153 0.91073254]
```

#### 41. How to sum a small array faster than np.sum? (★★☆) 
(**hint**: np.add.reduce)
```python
import numpy as np

a = np.arange(10)                                             # Array formation
b = np.add.reduce(a, dtype = int, axis = 0)                   # Reduction occurs column-wise with'int' datatype
  
print("The array {0} gets reduced to {1}".format(a, b))
```
Output:
```
The array [0 1 2 3 4 5 6 7 8 9] gets reduced to 45
```

#### 42. Consider two random array A and B, check if they are equal (★★☆) 
(**hint**: np.allclose, np.array\_equal)
```python
#Example
import numpy as np
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)
```
Output:
```
False
```

#### 43. Make an array immutable (read-only) (★★☆) 
(**hint**: flags.writeable)
```python
import numpy as np
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```
Output:
```
ValueError                                Traceback (most recent call last)

<ipython-input-44-aed29e88d03e> in <module>()
      2 Z = np.zeros(10)
      3 Z.flags.writeable = False
----> 4 Z[0] = 1


ValueError: assignment destination is read-only
```
  
#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 
(**hint**: np.sqrt, np.arctan2)
```python
import numpy as np

Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```
Output:
```
[0.68239687 0.94651921 0.70924704 0.65966959 0.67843111 1.05344618
 0.41228236 1.1917385  0.52601419 0.7976603 ]
[0.25451104 0.65369102 0.14674593 0.12347684 0.5366853  0.80220613
 0.49605296 0.94289808 1.48958449 1.31365005]
```
  
#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
(**hint**: argmax)
```python
import numpy as np

Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```
Output:
```
[0.92241937 0.7271441  0.         0.5763516  0.17485182 0.62590456
 0.86572222 0.08807759 0.0441941  0.90631484]
```

#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 
(**hint**: np.meshgrid)
```python
import numpy as np

Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                             np.linspace(0,1,10))
print(Z)
```
Output:
```
[[(0.        , 0.        ) (0.11111111, 0.        )
  (0.22222222, 0.        ) (0.33333333, 0.        )
  (0.44444444, 0.        ) (0.55555556, 0.        )
  (0.66666667, 0.        ) (0.77777778, 0.        )
  (0.88888889, 0.        ) (1.        , 0.        )]
 [(0.        , 0.11111111) (0.11111111, 0.11111111)
  (0.22222222, 0.11111111) (0.33333333, 0.11111111)
  (0.44444444, 0.11111111) (0.55555556, 0.11111111)
  (0.66666667, 0.11111111) (0.77777778, 0.11111111)
  (0.88888889, 0.11111111) (1.        , 0.11111111)]
 [(0.        , 0.22222222) (0.11111111, 0.22222222)
  (0.22222222, 0.22222222) (0.33333333, 0.22222222)
  (0.44444444, 0.22222222) (0.55555556, 0.22222222)
  (0.66666667, 0.22222222) (0.77777778, 0.22222222)
  (0.88888889, 0.22222222) (1.        , 0.22222222)]
 [(0.        , 0.33333333) (0.11111111, 0.33333333)
  (0.22222222, 0.33333333) (0.33333333, 0.33333333)
  (0.44444444, 0.33333333) (0.55555556, 0.33333333)
  (0.66666667, 0.33333333) (0.77777778, 0.33333333)
  (0.88888889, 0.33333333) (1.        , 0.33333333)]
 [(0.        , 0.44444444) (0.11111111, 0.44444444)
  (0.22222222, 0.44444444) (0.33333333, 0.44444444)
  (0.44444444, 0.44444444) (0.55555556, 0.44444444)
  (0.66666667, 0.44444444) (0.77777778, 0.44444444)
  (0.88888889, 0.44444444) (1.        , 0.44444444)]
 [(0.        , 0.55555556) (0.11111111, 0.55555556)
  (0.22222222, 0.55555556) (0.33333333, 0.55555556)
  (0.44444444, 0.55555556) (0.55555556, 0.55555556)
  (0.66666667, 0.55555556) (0.77777778, 0.55555556)
  (0.88888889, 0.55555556) (1.        , 0.55555556)]
 [(0.        , 0.66666667) (0.11111111, 0.66666667)
  (0.22222222, 0.66666667) (0.33333333, 0.66666667)
  (0.44444444, 0.66666667) (0.55555556, 0.66666667)
  (0.66666667, 0.66666667) (0.77777778, 0.66666667)
  (0.88888889, 0.66666667) (1.        , 0.66666667)]
 [(0.        , 0.77777778) (0.11111111, 0.77777778)
  (0.22222222, 0.77777778) (0.33333333, 0.77777778)
  (0.44444444, 0.77777778) (0.55555556, 0.77777778)
  (0.66666667, 0.77777778) (0.77777778, 0.77777778)
  (0.88888889, 0.77777778) (1.        , 0.77777778)]
 [(0.        , 0.88888889) (0.11111111, 0.88888889)
  (0.22222222, 0.88888889) (0.33333333, 0.88888889)
  (0.44444444, 0.88888889) (0.55555556, 0.88888889)
  (0.66666667, 0.88888889) (0.77777778, 0.88888889)
  (0.88888889, 0.88888889) (1.        , 0.88888889)]
 [(0.        , 1.        ) (0.11111111, 1.        )
  (0.22222222, 1.        ) (0.33333333, 1.        )
  (0.44444444, 1.        ) (0.55555556, 1.        )
  (0.66666667, 1.        ) (0.77777778, 1.        )
  (0.88888889, 1.        ) (1.        , 1.        )]]
```

####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 
(**hint**: np.subtract.outer)
```python
import numpy as np

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```
Output:
```
3638.1636371179666
```
  
#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 
(**hint**: np.iinfo, np.finfo, eps)
```python
import numpy as np

for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```
Output:
```
-128
127
-2147483648
2147483647
-9223372036854775808
9223372036854775807
-3.4028235e+38
3.4028235e+38
1.1920929e-07
-1.7976931348623157e+308
1.7976931348623157e+308
2.220446049250313e-16
```

#### 49. How to print all the values of an array? (★★☆) 
(**hint**: np.set\_printoptions)
```python
import numpy as np

np.set_printoptions(threshold=float("inf"))
X = np.zeros((40,40))
print(X)
```
Output:
```
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
 ```

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 
(**hint**: argmin)
```python
import numpy as np

X = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(X-v)).argmin()
print(X[index])
```
Output:
```
#random
67
```

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 
(**hint**: dtype)
```python
import numpy as np

Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(Z)
```
Output:
```
[((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
 ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
 ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
 ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
 ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))]
FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  Z = np.zeros(10, [ ('position', [ ('x', float, 1),
```

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 
(**hint**: np.atleast\_2d, T, np.sqrt)
```python
import numpy as np

Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)
  
import scipy                                              # Much faster with scipy
import scipy.spatial                                      # Thanks Gavin Heverly-Coulson (#issue 1)
Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```
Output:
```
[[0.         0.6837053  0.87693457 0.69932163 0.84089316 0.13772494
  0.33252113 0.47583549 0.78186516 0.9228082 ]
 [0.6837053  0.         1.09571332 0.52998708 0.22848634 0.81840347
  0.39382338 0.54179151 0.50981651 0.86531753]
 [0.87693457 1.09571332 0.         0.60376085 1.05478916 0.92200082
  0.80578567 0.56786752 0.68382887 0.39441623]
 [0.69932163 0.52998708 0.60376085 0.         0.45182035 0.81952342
  0.41608608 0.23426419 0.10503954 0.33721756]
 [0.84089316 0.22848634 1.05478916 0.45182035 0.         0.978611
  0.51508189 0.56194582 0.38935771 0.75705346]
 [0.13772494 0.81840347 0.92200082 0.81952342 0.978611   0.
  0.46920999 0.58946602 0.90689788 1.01690809]
 [0.33252113 0.39382338 0.80578567 0.41608608 0.51508189 0.46920999
  0.         0.25354058 0.478215   0.70932461]
 [0.47583549 0.54179151 0.56786752 0.23426419 0.56194582 0.58946602
  0.25354058 0.         0.33159788 0.4650036 ]
 [0.78186516 0.50981651 0.68382887 0.10503954 0.38935771 0.90689788
  0.478215   0.33159788 0.         0.36847191]
 [0.9228082  0.86531753 0.39441623 0.33721756 0.75705346 1.01690809
  0.70932461 0.4650036  0.36847191 0.        ]]
[[0.         0.54559758 0.76357724 0.61884646 0.73788943 0.79647129
  0.09169366 0.45023127 0.68757631 0.64729322]
 [0.54559758 0.         0.71228975 1.07012953 1.10046618 0.64907838
  0.45595018 0.60732169 0.94780871 0.82389458]
 [0.76357724 0.71228975 0.         0.8025797  0.68073149 0.13155542
  0.74425372 0.33321063 0.45940883 0.30959832]
 [0.61884646 1.07012953 0.8025797  0.         0.22456273 0.91484595
  0.69556555 0.5464748  0.38381322 0.50086168]
 [0.73788943 1.10046618 0.68073149 0.22456273 0.         0.80617832
  0.79801226 0.50954909 0.22188121 0.37286132]
 [0.79647129 0.64907838 0.13155542 0.91484595 0.80617832 0.
  0.76238316 0.40635912 0.58620434 0.43336174]
 [0.09169366 0.45595018 0.74425372 0.69556555 0.79801226 0.76238316
  0.         0.45243198 0.72534239 0.66626692]
 [0.45023127 0.60732169 0.33321063 0.5464748  0.50954909 0.40635912
  0.45243198 0.         0.34048753 0.22925303]
 [0.68757631 0.94780871 0.45940883 0.38381322 0.22188121 0.58620434
  0.72534239 0.34048753 0.         0.15507333]
 [0.64729322 0.82389458 0.30959832 0.50086168 0.37286132 0.43336174
  0.66626692 0.22925303 0.15507333 0.        ]]
```

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 
(**hint**: astype(copy=False))
```python
import numpy as np
Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)
```
Output:
```
#random
[49 52 49 87 18 63 86 63 18 30]
```

#### 54. How to read the following file? (★★☆) 
(**hint**: np.genfromtxt)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
```python
import numpy as np
from io import StringIO

s = StringIO('''1, 2, 3, 4, 5                      
                6,  ,  , 7, 8
                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```
Output:
```
[[ 1  2  3  4  5]
 [ 6 -1 -1  7  8]
 [-1 -1  9 10 11]]
DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
```

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 
(**hint**: np.ndenumerate, np.ndindex)
```python
import numpy as np

Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```
Output:
```
(0, 0) 0
(0, 1) 1
(0, 2) 2
(1, 0) 3
(1, 1) 4
(1, 2) 5
(2, 0) 6
(2, 1) 7
(2, 2) 8
(0, 0) 0
(0, 1) 1
(0, 2) 2
(1, 0) 3
(1, 1) 4
(1, 2) 5
(2, 0) 6
(2, 1) 7
(2, 2) 8
```

#### 56. Generate a generic 2D Gaussian-like array (★★☆) 
(**hint**: np.meshgrid, np.exp)
```python
import numpy as np

X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
Z = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(Z)
```
Output:
```
[[0.36787944 0.44822088 0.51979489 0.57375342 0.60279818 0.60279818
  0.57375342 0.51979489 0.44822088 0.36787944]
 [0.44822088 0.54610814 0.63331324 0.69905581 0.73444367 0.73444367
  0.69905581 0.63331324 0.54610814 0.44822088]
 [0.51979489 0.63331324 0.73444367 0.81068432 0.85172308 0.85172308
  0.81068432 0.73444367 0.63331324 0.51979489]
 [0.57375342 0.69905581 0.81068432 0.89483932 0.9401382  0.9401382
  0.89483932 0.81068432 0.69905581 0.57375342]
 [0.60279818 0.73444367 0.85172308 0.9401382  0.98773022 0.98773022
  0.9401382  0.85172308 0.73444367 0.60279818]
 [0.60279818 0.73444367 0.85172308 0.9401382  0.98773022 0.98773022
  0.9401382  0.85172308 0.73444367 0.60279818]
 [0.57375342 0.69905581 0.81068432 0.89483932 0.9401382  0.9401382
  0.89483932 0.81068432 0.69905581 0.57375342]
 [0.51979489 0.63331324 0.73444367 0.81068432 0.85172308 0.85172308
  0.81068432 0.73444367 0.63331324 0.51979489]
 [0.44822088 0.54610814 0.63331324 0.69905581 0.73444367 0.73444367
  0.69905581 0.63331324 0.54610814 0.44822088]
 [0.36787944 0.44822088 0.51979489 0.57375342 0.60279818 0.60279818
  0.57375342 0.51979489 0.44822088 0.36787944]]
```

#### 57. How to randomly place p elements in a 2D array? (★★☆) 
(**hint**: np.put, np.random.choice)
```python
import numpy as np

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print (Z)
```
Output:
```
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```
  
#### 58. Subtract the mean of each row of a matrix (★★☆) 
(**hint**: mean(axis=,keepdims=))
```python
import numpy as np

X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)             # Recent versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)             # Older versions of numpy
print(Y)
```
Output:
```
[[-0.46647919 -0.22785805  0.47552928  0.3984473  -0.41337391 -0.27749467
   0.25668582 -0.23524927  0.08449316  0.40529953]
 [-0.1612853  -0.02727661 -0.05212843  0.49526687  0.03646314  0.56110108
   0.04172705 -0.17430362 -0.28464718 -0.434917  ]
 [-0.00964443  0.35466582 -0.29911968 -0.10610349 -0.14606521 -0.22172862
   0.07389907  0.62728508 -0.11256618 -0.16062237]
 [ 0.15669637 -0.13853239 -0.20639963 -0.11677556 -0.24068708  0.59644536
  -0.35415446  0.10503397 -0.36591373  0.56428715]
 [-0.15147641  0.25513489 -0.51472191  0.41859795 -0.04248227  0.22101292
  -0.33749177  0.38515417  0.28819665 -0.52192424]]
```

#### 59. How to sort an array by the nth column? (★★☆) 
(**hint**: argsort)
```python
import numpy as np

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```
Output:
```
#random
[[0 8 8]
 [7 6 6]
 [1 1 7]]
[[1 1 7]
 [7 6 6]
 [0 8 8]]
```

#### 60. How to tell if a given 2D array has null columns? (★★☆) 
(**hint**: any, ~)
```python
import numpy as np

Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())
```
Output:
```
False
```
  
#### 61. Find the nearest value from a given value in an array (★★☆) 
(**hint**: np.abs, argmin, flat)
```python
import numpy as np

Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```
Output:
```
#random
0.5305518156479256
```

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 
(**hint**: np.nditer)
```python
import numpy as np

A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```
Output:
```
[[0 1 2]
 [1 2 3]
 [2 3 4]]
```

#### 63. Create an array class that has a name attribute (★★☆) 
(**hint**: class method)
```python
import numpy as np
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', "no name")

X = NamedArray(np.arange(10), "range_10")
print (X.name)
```
Output:
```
range_10
```

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 
(**hint**: np.bincount | np.add.at)
```python
import numpy as np
  
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)
```
Output:
```
#random
[3. 6. 2. 2. 4. 2. 1. 3. 3. 4.]
```
#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 
(**hint**: np.bincount)
```python
import numpy as np
  
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```
Output:
```
[0. 7. 0. 6. 5. 0. 0. 0. 0. 3.]
```

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 
(**hint**: np.unique)
```python
import numpy as np

w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))
```
Output:
```
[0 1]
```

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 
(**hint**: sum(axis=(-2,-1)))
```python
import numpy as np

A = np.random.randint(0,10,(3,4,3,4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```
Output:
```
#random
[[46 60 60 59]
 [58 51 54 59]
 [44 42 45 45]]
```
  
#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 
(**hint**: np.bincount)
```python
import numpy as np

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)
```
Output:
```
#random
[0.41727248 0.42913527 0.7179096  0.54933571 0.46669399 0.4673405
 0.4418931  0.5549945  0.61780927 0.66975341]
```

#### 69. How to get the diagonal of a dot product? (★★★) 
(**hint**: np.diag)
```python
import numpy as np

A = np.random.randint(0,10,(3,3))
B= np.random.randint(0,10,(3,3))

np.diag(np.dot(A, B))                    # Slow version
np.sum(A * B.T, axis=1)                  # Fast version
np.einsum("ij,ji->i", A, B)              # Faster version
```
Output:
```
#random
array([27, 93, 16])
```

#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
(**hint**: array\[::4\])
```python
import numpy as np

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```
Output:
```
[1. 0. 0. 0. 2. 0. 0. 0. 3. 0. 0. 0. 4. 0. 0. 0. 5.]
```
  
#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
(**hint**: array\[:, :, None\])
```python
import numpy as np

A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```
Output:
```
[[[2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]]

 [[2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]]

 [[2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]]

 [[2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]]

 [[2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]
  [2. 2. 2.]]]
```
  
#### 72. How to swap two rows of an array? (★★★) 
(**hint**: array\[\[\]\] = array\[\[\]\])
```python
import numpy as np

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```
Output:
```
[[ 5  6  7  8  9]
 [ 0  1  2  3  4]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
```

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 
(**hint**: repeat, np.roll, np.sort, view, np.unique)
```python
import numpy as np

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```
Output:
```
[(11, 51) (11, 98) (15, 30) (15, 59) (15, 62) (15, 64) (15, 69) (15, 89)
 (17, 23) (17, 95) (23, 95) (24, 58) (24, 93) (30, 62) (32, 80) (32, 89)
 (48, 54) (48, 81) (50, 54) (50, 59) (51, 98) (54, 59) (54, 81) (58, 93)
 (59, 89) (64, 69) (70, 80) (80, 80) (80, 89)]
```

#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 
(**hint**: np.repeat)
```python
import numpy as np

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```
Output:
```
[1 1 2 3 4 4 6]
```

#### 75. How to compute averages using a sliding window over an array? (★★★) 
(**hint**: np.cumsum)
```python
import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
```
Output:
```
[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]
```
  
#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 
(**hint**: from numpy.lib import stride_tricks)
```python
import numpy as np

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```
Output:
```
[[0 1 2]
 [1 2 3]
 [2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]
 [6 7 8]
 [7 8 9]]
```
  
#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 
(**hint**: np.logical_not, np.negative)
```python
import numpy as np

Z = np.random.randint(0,2,100)
print ('original: ')
print (Z)
print('Negating a boolean: ')
print(np.logical_not(Z, out=Z))


Z = np.random.uniform(-1.0,1.0,10)
print ('original: ')
print (Z)
print ('Change the sign of float inplace: ')
print(np.negative(Z, out=Z))
```
Output:
```
original: 
[1 1 1 1 0 1 0 1 1 0 0 0 1 0 1 1 0 1 1 0 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1 0 1
 1 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 1 1 0 1 0 0 0 0 0 0 1 0 1 1 1 0 0 1 0 1
 1 1 1 1 0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 0 1 0]
Negating a boolean: 
[0 0 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0
 0 0 1 0 0 1 1 1 1 1 1 1 0 1 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 0 0 0 1 1 0 1 0
 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 1]
original: 
[-0.06836599  0.04793971  0.6589628  -0.94529021 -0.87511555 -0.36042867
 -0.25548385  0.42935055 -0.99462483 -0.89615931]
Change the sign of float inplace: 
[ 0.06836599 -0.04793971 -0.6589628   0.94529021  0.87511555  0.36042867
  0.25548385 -0.42935055  0.99462483  0.89615931]
```

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)
```python
import numpy as np

def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```
Output:
```
#random
[ 1.48187788  0.76219602  5.13748819  2.23377232  0.06870344  0.19233972
  2.76808007 12.28279927  5.76718417  1.2419688 ]
```

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)
```python
import numpy as np

P0 = np.random.uniform(-10, 10, (5,2))
P1 = np.random.uniform(-10,10,(5,2))
p = np.random.uniform(-10, 10, (5,2))
print (np.array([distance(P0,P1,p_i) for p_i in p]))
```
Output:
```
#random
[[ 1.13801647  3.94641267  9.28238157  3.21996078  0.87552165]
 [ 1.58216632  3.86949657  8.94877174  2.7613081   0.57445507]
 [12.99729203  9.99095316  4.98460306  1.46770069  6.92546331]
 [15.55367254  5.30865891  4.59349335  5.69796484  0.88336616]
 [ 2.6255866   7.39170491  9.83496375  1.5861636   4.52283845]]
```

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 
(**hint**: minimum, maximum)
```python
import numpy as np

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```
Output:
```
[[1 5 2 1 6 4 5 4 0 6]
 [8 3 2 6 1 2 2 3 7 8]
 [1 6 0 6 5 0 1 5 4 4]
 [4 6 5 7 6 4 7 1 0 3]
 [1 2 5 8 1 9 9 2 2 1]
 [4 4 4 4 3 2 8 1 5 2]
 [0 0 4 2 7 8 1 3 1 1]
 [6 6 0 8 8 2 2 7 0 5]
 [2 8 6 1 1 3 6 8 0 4]
 [4 9 7 0 4 0 2 5 3 5]]
[[0 0 0 0 0]
 [0 1 5 2 1]
 [0 8 3 2 6]
 [0 1 6 0 6]
 [0 4 6 5 7]]
FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  R[r] = Z[z]

#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 
(**hint**: stride\_tricks.as\_strided)
```python
import numpy as np

Z = np.arange(1,15,dtype=int)

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
R = rolling(Z, 4)
print ('original: ')
print (Z)
print ('after strides: ')
print(R)
```
Output:
```
original: 
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]
after strides: 
[[ 1  2  3  4]
 [ 2  3  4  5]
 [ 3  4  5  6]
 [ 4  5  6  7]
 [ 5  6  7  8]
 [ 6  7  8  9]
 [ 7  8  9 10]
 [ 8  9 10 11]
 [ 9 10 11 12]
 [10 11 12 13]
 [11 12 13 14]]
```
  
#### 82. Compute a matrix rank (★★★) 
(**hint**: np.linalg.svd) (suggestion: np.linalg.svd)
```python
import numpy as np

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print (rank)
```
Output:
```
10
```

#### 83. How to find the most frequent value in an array? 
(**hint**: np.bincount, argmax)
```python
import numpy as np

Z = np.random.randint(0,10,50)
print (Z)
print('rank:', np.bincount(Z).argmax())
```
Output:
```
#random
[0 8 3 2 1 1 6 6 8 6 5 3 1 3 6 8 1 5 9 9 6 6 1 5 8 0 2 2 1 1 0 7 2 7 0 6 2
 8 9 8 6 1 3 6 1 3 8 4 0 5]
rank: 1
```
  
#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 
(**hint**: stride\_tricks.as\_strided)
```python
import numpy as np

Z = np.random.randint(0,5,(6,6))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = np.lib.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)
```
Output:
#random
[[[[3 4 3]
   [4 4 4]
   [0 4 4]]

  [[4 3 3]
   [4 4 4]
   [4 4 4]]

  [[3 3 0]
   [4 4 0]
   [4 4 3]]

  [[3 0 0]
   [4 0 2]
   [4 3 3]]]


 [[[4 4 4]
   [0 4 4]
   [1 1 2]]

  [[4 4 4]
   [4 4 4]
   [1 2 3]]

  [[4 4 0]
   [4 4 3]
   [2 3 0]]

  [[4 0 2]
   [4 3 3]
   [3 0 0]]]


 [[[0 4 4]
   [1 1 2]
   [0 1 3]]

  [[4 4 4]
   [1 2 3]
   [1 3 4]]

  [[4 4 3]
   [2 3 0]
   [3 4 2]]

  [[4 3 3]
   [3 0 0]
   [4 2 0]]]


 [[[1 1 2]
   [0 1 3]
   [2 1 2]]

  [[1 2 3]
   [1 3 4]
   [1 2 3]]

  [[2 3 0]
   [3 4 2]
   [2 3 2]]

  [[3 0 0]
   [4 2 0]
   [3 2 0]]]]
```
#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 
(**hint**: class method)
```python
import numpy as np

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```
Output:
```
[[ 1  8 14 11 12]
 [ 8  0  9  5 10]
 [14  9  2 42  9]
 [11  5 42  6 12]
 [12 10  9 12  2]]
```

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 
(**hint**: np.tensordot)
```python




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
