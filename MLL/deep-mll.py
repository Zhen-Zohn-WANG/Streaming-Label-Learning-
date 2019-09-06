# -*- coding:utf-8 -*- 

# Author:  Zhen Wang

# DATE:    2019/1/1   

# Time:    下午6:41  

# IDE:     PyCharm Community Edition

# For Deep Multi-label Learning

import tensorflow as tf
from numpy.random import  RandomState

# labels
m = 500
# size
n = 10000
# new labels
k = 10

INPUT_NODE = m * n
OUTPUT_NODE = m
LAYER1_NODE = m * n
BATCH_SIZE = 100



LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):

    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.nn.relu(tf.matmul(layer1, weights2) + biases2)

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train():
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # Hidden
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # Output
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 不含滑动平均类的前向传播
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # Training step
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算mse
    mse_loss = tf.reduce_mean(tf.square(y_ - y))

    # Loss function
    regularizer = tf.contrib.layers.l1_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = mse_loss + regularaztion

    rdm = RandomState(1)
    dataset_size = n
    validate_size = n/10
    test_size = n/5
    X = rdm.rand(dataset_size+validate_size+test_size,m)
    X_train = X[:][0:dataset_size]
    X_validate = X[:][dataset_size:dataset_size+validate_size]
    X_test = X[:][dataset_size+validate_size:dataset_size+validate_size+test_size]
    y_ = [[x1+x2+rdm.rand()/10-0.05] for (x1,x2) in X]
    y_train = y_[0:dataset_size]
    y_validate = y_[dataset_size:dataset_size+validate_size]
    y_test = y_[dataset_size+validate_size:dataset_size+validate_size+test_size]
    BATCH_SIZE = 1

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        dataset_size / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # Optimize
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    train_op = tf.group(train_step, variables_averages_op)

    # correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.reduce_mean(tf.square(y_ - y))


    # Initial
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: X_train, y_: y_train}
        test_feed = {x: X_test, y_: y_test}

        # Training
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs = X_train[:][i:(i+1)*BATCH_SIZE]
            ys = y_train[i:(i+1)*BATCH_SIZE]
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))


def main(argv=None):
    train()

if __name__ == '__main__':
    main()


