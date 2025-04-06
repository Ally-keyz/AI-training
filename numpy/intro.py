#importing of modules
import numpy as np

a = np.array([[1,2,3]
             ,[4,5,6]])
print(f"This is a shape: {a.shape}")

# exploiting arrays in numpy

b = np.array([1,2,3,4,5,6,7])

#accessing indexes of arrays
print(b[0])
print(b[:4])

c = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(c[1,2])
print(c[2,1])
print(c[0,0])
print(f"dimension : {c.ndim}")
print(f"shape : {c.shape}")
print(f"size : {c.size}")
print(f"datatype : {c.dtype}")

#sorting arrays

d = np.array([2,5,6,8,1,9])
sorted = np.sort(d)
print(sorted)

#reshaping an array
e = np.array([9,8,7,6,5,4])
print(np.arange(6))
print(np.reshape(d,shape=(3,2),order="C"))