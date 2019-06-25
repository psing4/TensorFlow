
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784])
 #placeholder -> 784개의 배열 값 장소를 만듬 (none은 데이터 겟수 미정)
W = tf.Variable(tf.zeros([784, 10]))
#Variable 784개의 값을 가지고 claas 10를 나누고 0으로 채움 
b = tf.Variable(tf.zeros([10]))
#Variable 0값을 집어넣음 
y = tf.nn.softmax(tf.matmul(x, W) + b)
#softmax -> 결과를 전체 합계가 1이 되는 0과 1 사이의 값으로 변경(=확률)
#matmul은 행렬 곱 
#y는 output값 


y_ = tf.placeholder(tf.float32, [None, 10])
#y_는 정답 값 10갸의 클래스로 나누어져 있다.

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#정보 이득 정리의 교차 엔트로피 정보를 나타낼때 필요한 비트수를 의미 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#경사하강 최적화를 이용하여 0.01의 기울기 변경을 통해 크로스 엔트로피를 최소화 한다.

# Session
init = tf.global_variables_initializer()
#초기화 

sess = tf.Session() #세션을 연다.
sess.run(init) #세션을 초기화


# Learning
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #next_batch 데이터셋으로부터 필요한 만큼의 데이터를 반환하는 함수
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  #feed_dict x의 값에 batch_xs값을 하나씩 넣는다 y_도 동일 
  #x값은 y값을 조정한다. 


# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
"""
one-hot(원핫)인코딩이란? 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말한다.
즉, 1개만 Hot(True)이고 나머지는 Cold(False)이다.

1) 원핫인코딩된 데이터중 가장 큰 값은 당연히 "1"일 것이고 해당 (Zero-based)인덱스를 리턴한다.

2) Softmax를 통해 나온 결과중 최대값의 인덱스를 얻고자 할 때 사용한다.

참고로 최대 값이 2개 이상인 경우 더 앞선 인덱스 값이 나온다.

"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
