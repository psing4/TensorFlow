# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 20 #20번 돌리겠다.
batch_size = 200 #200개를

# input place holders
X = tf.placeholder(tf.float32, [None, 784]) #한줄의 행렬로 나타냄 
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)이런걸로 변환 1은 단일 색상을 의미함 
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) #3*3필터를 가지고 32개의 필터를 가지고 랜덤하게 생성 
#    Conv     -> (?, 28, 28, 32) -> 28를 3X3으로 행렬곱으로 처리해서 32개를 쌓는다.
#    Pool     -> (?, 14, 14, 32) -> 2 X 2를 행렬 중 가장 큰갑 하나로 줄인다. 그래서 14 X 14로 줄음 
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') #strides는 1,1,1,1 은 한칸씩, padding크기는 같게 
'''Convolution 레이어에서 Filter와 Stride에 작용으로 Feature Map 크기는 입력데이터 보다 작습니다. 
Convolution 레이어의 출력 데이터가 줄어드는 것을 방지하는 방법이 패딩입니다. 
패딩은 입력 데이터의 외각에 지정된 픽셀만큼 특정 값으로 채워 넣는 것을 의미합니다. 
보통 패딩 값으로 0으로 채워 넣습니다.'''

L1 = tf.nn.relu(L1) #활성화 함수 에 넣는다 *어떤값 이하는 0으로 취급 하는 함수 * 최솟값만 
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME') #ksize=[1, 2, 2, 1] 2x2의 형태 
#max_pool은 2X2의 필터를 통해 가장 큰값을 뽑아낸다. 
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) #32개를 64개로 바꿈
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')


# L2 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)) # 64개를 128로 바꿈
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 1, 1, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
L3_flat = tf.reshape(L3, [-1, 7 * 7 * 128]) #6272 노드로 바꿈 


# Final FC 4 * 4 * 128 inputs -> 10 outputs 
W4 = tf.get_variable("W4", shape=[7 * 7 * 128, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_flat, W4) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Learning started. It takes sometime.
Epoch: 0001 cost = 0.464334811
Epoch: 0002 cost = 0.081364721
Epoch: 0003 cost = 0.056179016
Epoch: 0004 cost = 0.042146173
Epoch: 0005 cost = 0.032512850
Epoch: 0006 cost = 0.028192955
Epoch: 0007 cost = 0.024411595
Epoch: 0008 cost = 0.021153035
Epoch: 0009 cost = 0.017840849
Epoch: 0010 cost = 0.014973885
Epoch: 0011 cost = 0.013977723
Epoch: 0012 cost = 0.010523610
Epoch: 0013 cost = 0.010040470
Epoch: 0014 cost = 0.009902531
Epoch: 0015 cost = 0.007296389
Epoch: 0016 cost = 0.007485737
Epoch: 0017 cost = 0.007936805
Epoch: 0018 cost = 0.006387888
Epoch: 0019 cost = 0.005561064
Epoch: 0020 cost = 0.006455854
Learning Finished!
Accuracy: 0.9906
Label:  [1]
Prediction:  [1]
'''

