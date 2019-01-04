import tensorflow as tf
import numpy as np
import data_help
import config

def add_lay(input, in_size, out_size, activate_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            weight = tf.Variable(tf.random_normal([in_size, out_size],name='W'))
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]),name='b')
        with tf.name_scope('xw_plus_b'):
            xw_plus_b = tf.matmul(input, weight) + bias

        if(activate_function == None):
            out_put = xw_plus_b
        else:
            out_put = (activate_function(xw_plus_b))
        return out_put



def softmax_model():
    # 数据输入
    dh = data_help.DataHelper('w2v_60.csv')
    print('Data loaded...')
    # dh.get_data()
    dh.split_train_validation_test()
    print('Data splited...')

    with tf.name_scope('In_put'):
        x = tf.placeholder(tf.float32, shape=[None, 60], name='x_placeholder')
        y = tf.placeholder(tf.float32, shape=[None, 9], name='y_placeholder')

    # with tf.name_scope('Infrence'):
    #     weight = tf.Variable(tf.random_normal(shape=[60, 9]), name='weight')
    #     bias = tf.Variable(initial_value=tf.zeros(shape=[9]), name='bias')
    #     Wx_plus_b = tf.matmul(x, weight) + bias
    #     y_prediction = tf.nn.softmax(Wx_plus_b)
    #layer1 = add_lay(x, 60, 30, activate_function=tf.nn.tanh)
    #layer2 = add_lay(layer1, 30, 30, activate_function=tf.nn.tanh)
    y_prediction = add_lay(x, 60, 9, activate_function=tf.nn.softmax)


    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_prediction, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')

    with tf.name_scope('Optimization'):
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.name_scope('Evaluate'):
        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('~~~~~~~~~~~开始执行计算图~~~~~~~~~~~~~~')
    saver = tf.train.Saver(max_to_keep=3)# 保存训练的中间模型
    n_batches = int(dh.train_length / config.FLAGS.train_batch_size)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir=config.FLAGS.log_dir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        for i in range(0, config.FLAGS.epoch):
            for j in range(0, n_batches):
                x_batch, y_batch = dh.next_batch(config.FLAGS.train_batch_size)
                _, loss_batch = sess.run([train_op, loss], feed_dict={x: x_batch, y: y_batch})
                total_loss += loss_batch
                if j % config.FLAGS.disp_freq == 0:
                    val_acc = sess.run(accuracy, feed_dict={x: dh.val_x, y: dh.val_y})
                    if j == 0:
                        print('epoch :{}, step: {}, train_loss: {}, val_acc: {}'.format(i, j, total_loss, val_acc))
                    else:
                        print('epoch :{}, step: {}, train_loss: {}, val_acc: {}'.format(i, j, total_loss / config.FLAGS.disp_freq, val_acc))
                    total_loss = 0
            dh.batch_sum = 0
            saver.save(sess, 'ckpt/classfication_softmax.ckpt', global_step=i + 1)
        test_acc = sess.run(accuracy, feed_dict={x: dh.test_x, y: dh.test_y})
        print('test accuracy: {}'.format(test_acc))
        summary_writer.close()


if __name__ == "__main__":
    print('Start...')
    softmax_model()
