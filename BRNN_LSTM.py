import pandas as pd
import numpy as np
import tensorflow as tf

learning_rate = 0.01
learning_rate_delay = 0.96
batch_size = 100
max_steps = 30000

def calculate_loss(y_, y_prediction):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_prediction))


# train_step = tf.Variable(0, trainable=False)
# loss = tf.Variable(tf.truncated_normal([3, 10], stddev=0.1))
# lr = tf.train.exponential_decay(0.001, train_step,
#                                 1000, 0.96, staircase=True)
# optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=train_step)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     for i in range(10000):
#         train_step = i
#         sess.run(optimizer)
#         print('lr=', sess.run(lr))

# fc1_out = [[0, 2, 1],
#            [0, 1, 0],
#            [0, 1, 0]]
#
# fc2_out = [[0, 0, 1],
#            [0, 1, 0],
#            [0, 1, 0]]
#
# co = tf.equal(tf.argmax(fc1_out, 1), tf.argmax(fc2_out, 1))
#
# test = []
# test.append(fc1_out)
# test.append(fc2_out)
# print(test)
# tt = tf.concat(test, 1)
#
#
#
# value = list([[[0., 0.85, 0.15], [0, 0.99, 0.01]],
#                     [[0.1, 0.9, 0], [0.01, 0.99, 0]],
#                     [[0., 0.8, 0.2], [0, 0.9, 0.1]]])
#
# tt1 = tf.concat(value, 1)
#
#
# label = np.asarray([
#                     [[0., 1., 0], [0, 1, 0]],
#                     [[1., 0., 0], [0, 1, 0]],
#                     [[1., 0., 0], [0, 1, 0]]
#                     ])
# loss = calculate_loss(label, value)
# correct = tf.equal(tf.argmax(value, 2), tf.argmax(label, 2))
# correct = tf.cast(correct, tf.float32)
# avg_ = tf.reduce_mean(correct, axis=1)
# one = tf.ones(tf.shape(avg_))
# accuracy = tf.reduce_mean(tf.cast(tf.equal(avg_, one), tf.float32))
# #print(label[:])
# #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=value))
# #for i in range(label.shape[0]):
# max1 = tf.argmax(label[0], 1)
#
# #cor = tf.Variable(label[0], dtype=tf.float32)
# cor = tf.equal(tf.argmax(label, 2), tf.argmax(value, 2))
# cor = tf.cast(cor, tf.float32)
# avg = tf.reduce_mean(cor, axis=1)
# len = tf.shape(avg)
# o = tf.zeros(len)
# t = tf.equal(avg, o)
# t = tf.reduce_mean(tf.cast(t, tf.float32))
# # for i in range(label.shape[2]):
# #     pass
#     #cor = tf.argmax(label[:, :, i], 0)
#     #cor = tf.equal(tf.argmax(label[:, :, i], 1), tf.argmax(value[:, :, i], 1))
#     #cor = tf.cast(cor, tf.float32)
# #cor_ = tf.reduce_sum()
#
#
# #print(label.shape[0])
# #print(label[:, 1, :])
# #correct1 = tf.equal(tf.argmax(fc1_out, 1), tf.argmax(label[:, 0, :], 1))
# #correct2 = tf.equal(tf.argmax(fc2_out, 1), tf.argmax(label[:, 1, :], 1))
# #correct = tf.equal(correct1, correct2)
# #correct1 = tf.cast(correct1, tf.float32)
# #correct2 = tf.cast(correct2, tf.float32)
# #correct = tf.multiply(correct1, correct2)
#
# #acc1 = tf.reduce_mean(tf.cast(correct1, dtype=tf.float32))
# #acc2 = tf.reduce_mean(tf.cast(correct2, dtype=tf.float32))
# #acc = tf.reduce_mean(correct)
# with tf.Session() as sess:
#     with sess.as_default():
#         print(sess.run(f1))
#         print(sess.run(tt1))
#         print(sess.run(accuracy))
#
#     #print(sess.run(loss))
#     # print(sess.run(correct1))
#     # print(sess.run(acc1))
#     # print(sess.run(correct2))
#     # print(sess.run(acc2))
#     # print(sess.run(correct))
#     # print(sess.run(acc))


x = tf.placeholder(tf.float32, shape=[None, 1])
y = 4 * x + 4

w = tf.Variable(tf.random_normal([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y_predict = w * x + b

loss = tf.reduce_mean(tf.square(y - y_predict))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

isTrain = False
train_steps = 100
checkpoint_steps = 50
checkpoint_dir = 'myModelG'

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    if isTrain:
        for i in range(train_steps):
            sess.run(train, feed_dict={x: x_data})
            if (i + 1) % checkpoint_steps == 0:
                # saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)
                saver.save(sess, 'myModelG/model.ckpt', global_step=i + 1)
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass
        print(sess.run(w))
        print(sess.run(b))
        value = 3 * w + b
        rrr = sess.run(value)
        print(rrr)
