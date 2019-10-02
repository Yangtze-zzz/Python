numpy

vector = numpy.array([5, 10, 15, 20])

matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40 , 45] ])

print(vector)	[ 5 10 15 20]
print(matrix)	[[ 5 10 15]
                                 [20 25 30]
                                 [35 40 45]]

vector = numpy.array([1, 2, 3, 4])
print(vector.shape)		(4,)  查看维度，几行几列

matrix = numpy.array([[5, 10, 15], [20, 25, 30]])
print(matrix.shape)		(2, 3)  两行三列

.dtype 查看当前类型

			文件名	分隔符	   读进来的类型   去掉第一行
content = numpy.genfromtxt('test.txt', delimiter=",", dtype=str, skip_header=1)

a = content[1, 4]  （2个都从从0开始计数）第1个样本的第四列


matrix = numpy.array([
		   [5, 10, 15],
	   	   [20, 25, 30],
		   [35, 40 ,45]
		  ])
print(matrix[:, 1])		取中间一列，冒号表示占位	
[10 25 40]

取后两列
print(matrix[:, 0:2])


vector = numpy.array([5, 10, 15, 20])
vector == 10
Out：array([False,  True, False, False])

.astype(float)		转换为float类型

.min()
.max()
.sum(axis=1)		按行进行求和
.sum(axis=0)		按列进行求和

import numpy as np
print(np.arange(15))		[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
a=np.arange(15).reshape(3,5)
a

Out:array([[ 0,  1,  2,  3,  4],
      	 [ 5,  6,  7,  8,  9],
      	 [10, 11, 12, 13, 14]])

a.shape			当前矩阵多少行多少列	(3, 5) 
a.ndim			查看矩阵维度为多少		2
a.dytpe.name		查看类型			'int32'
a.size			查看有多少个元素		15

np.zeros((3, 4))

Out:array([[0., 0., 0., 0.],
   	[0., 0., 0., 0.],
       	[0., 0., 0., 0.]])

np.ones((2,3,4), dtype=np.int32)	创建了维度为3 x轴y轴z轴	值类型为int32
Out:array([[[1, 1, 1, 1],
        	 [1, 1, 1, 1],
       	 [1, 1, 1, 1]],
	
     	  [[1, 1, 1, 1],
      	  [1, 1, 1, 1],
      	  [1, 1, 1, 1]]])

np.arange(10, 30, 5)		从10开始到30结束 每次加5	array([10, 15, 20, 25])

np.random.random((2,3))	随机

from numpy import pi
np.linspace(0, 2*pi, 100)	从第一个数为0到最后一个数为2*pi中间找100个数平均的找  

a = np.array([20, 30, 40, 50])
b = np.arange( 4 )
print(a)
print(b)
c = a - b 
print(c)

Out:	[20 30 40 50]
	[0 1 2 3]
	[20 29 38 47]

对array中元素进行-1 会对所有元素都进行-1

A = np.array([[1,1],
	     [0,1]])
B = np.array([[2,0],
	     [3,4]])

print(A*B)

Out:[[2 0]
       [0 4]]

print(A.dot(B))	进行矩阵操作

Out：[[5 4]
          [3 4]]

print(np.dot(A,B))	进行矩阵相乘       和上一个都可以


Out：[[5 4]
          [3 4]]

np.exp(B)		B的e次幂
np.sqrt(B)		根号B


floor
np.floor 返回不大于输入参数的最大整数。 即对于输入值 x ，将返回最大的整数 i ，使得 i <= x。 注意在Python中，向下取整总是从 0 舍入。

a.ravel()		把矩阵变成一行
a.shape=(6,2)	把矩阵变成六行两列
np.vstack(a,b)	把矩阵按照行拼接在一起
np.hstack(a,b)	把矩阵按照列进行拼接

np.hsplit(a,3)	按照列切分每4个数切1下切3次

import numpy as np
a = np.floor(10*np.random.random((2,12)))
print(a)
print('---')
print(np.hsplit(a,3))

[[3. 0. 1. 3. 5. 6. 6. 5. 0. 6. 3. 4.]
 [5. 9. 4. 9. 6. 3. 3. 5. 9. 4. 4. 9.]]
---
[array([[3., 0., 1., 3.],
       [5., 9., 4., 9.]]), array([[5., 6., 6., 5.],
       [6., 3., 3., 5.]]), array([[0., 6., 3., 4.],
       [9., 4., 4., 9.]])]

np.hsplit(a,(3,4))	在3和4的位置切一刀

[[3. 0. 1. 3. 5. 6. 6. 5. 0. 6. 3. 4.]
 [5. 9. 4. 9. 6. 3. 3. 5. 9. 4. 4. 9.]]
[array([[3., 0., 1.],
       [5., 9., 4.]]), array([[3.],
       [9.]]), array([[5., 6., 6., 5., 0., 6., 3., 4.],
       [6., 3., 3., 5., 9., 4., 4., 9.]])]

np.vsplit   按列切

a = np.arange(12)
b = a
print(id(a))
print(id(b))
a 和b的地址是完全一样的 a == b
b改变还是a改变另一个同时也改变

c = a.view()         浅复制，变换c中的一个值，a中有一个值也会复制，地址不同，共用一组数据



a = np.arange(12)
a.shape = 3,4
print(a)
c=a.view()
c.shape = 2,6
print(a.shape)
c[0,4] = 1234
print(a)
print(id(a))
print(id(c))

[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
(3, 4)
[[   0    1    2    3]
 [1234    5    6    7]
 [   8    9   10   11]]
2314096506480
2314096505600

d = a.copy()		d是a的初始化，a改变后d不会改变

data = np.sin(np.arange(20)).reshape(5,4)
print(data)
ind = data.argmax(axis = 0)	返回每一列中的最大值的索引，axis=0 是在列中比较 axis=1 是在行中比较
print(ind)
data_max = data[ind, range(data.shape[1])]
print(data_max)

[[ 0.          0.84147098  0.90929743  0.14112001]
 [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
 [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
 [-0.53657292  0.42016704  0.99060736  0.65028784]
 [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
[2 0 3 1]
[0.98935825 0.84147098 0.99060736 0.6569866 ]




a = np.arange(0, 40, 10)
print(a)
b = np.tile(a, (4, 2))		行变成原来的4倍，列变成原来的2倍
print(b)

[ 0 10 20 30]
[[ 0 10 20 30  0 10 20 30]
 [ 0 10 20 30  0 10 20 30]
 [ 0 10 20 30  0 10 20 30]
 [ 0 10 20 30  0 10 20 30]]

a = np.array([[4, 3, 5], [1, 2, 1]])
print(a)
print('-----')
b = np.sort(a, axis=1)	按行进行排列
print(b)

[[4 3 5]
 [1 2 1]]
-----
[[3 4 5]
 [1 1 2]]


a=np.array([4, 3, 1, 2])
j = np.argsort(a)		#按索引值进行排序
print('-----')
print(j)
print('-----')
print(a[j])

-----
[2 3 1 0]
-----
[1 2 3 4]
