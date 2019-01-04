import data_help
import tensorflow as tf
import config
import numpy as np
#from tensorflow.python import debug as tf_debug

print(tf.__version__)

Multi_LSTM = True
numClasses = 9
num_step = 10
num_dim = 60
def LSTM_model():
    # 数据输入
    dh = data_help.DataHelper('F:\\Python\\NLP\\data\\radom_10w.csv')
    print('Data loaded...')
    dh.split_train_validation_test()
    print('Data splited...')

    data = tf.placeholder(dtype=tf.float32, shape=([None, num_step, num_dim]))
    label = tf.placeholder(dtype=tf.float32, shape=[None, numClasses])

    if Multi_LSTM:
        if config.FLAGS.hidden_layer_num > 1:
            lstmCell = [tf.contrib.rnn.LSTMCell(size) for size in [num_dim, config.FLAGS.lstm_hidden_units]]#config.FLAGS.lstm_hidden_units
            lstmCell = tf.contrib.rnn.MultiRNNCell(lstmCell, state_is_tuple=True)
        else:
            lstmCell = tf.contrib.rnn.LSTMCell(config.FLAGS.lstm_hidden_units)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #initial_state = lstmCell.zero_state(config.FLAGS.train_batch_size, dtype=tf.float32)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    else:
        lstmCell = tf.contrib.rnn.LSTMCell(config.FLAGS.lstm_hidden_units)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #initial_state = lstmCell.zero_state(config.FLAGS.train_batch_size, dtype=tf.float32)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    value = tf.transpose(value, [1, 0, 2])
    # 取最终的结果值
    last = tf.gather(value, int(value.shape[0]) - 1)
    weight = tf.Variable(tf.truncated_normal([config.FLAGS.lstm_hidden_units, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=label))
    optimizer = tf.train.AdamOptimizer(config.FLAGS.learning_rate).minimize(loss)

    num_batches = int(dh.train_length/config.FLAGS.train_batch_size)
    total_loss = 0
    saver = tf.train.Saver(max_to_keep=3)  # 保存训练的中间模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        for i in range(config.FLAGS.epoch):
            for j in range(num_batches):
                x_batch, y_batch = dh.next_batch(config.FLAGS.train_batch_size)
                _, batch_loss = sess.run([optimizer, loss], feed_dict={data: x_batch, label: y_batch})
                total_loss += batch_loss
                if j % config.FLAGS.disp_freq == 0:
                    acc = sess.run(accuracy, feed_dict={data: dh.val_x, label: dh.val_y})
                    if j == 0:
                        print('epoch:{},batch:{},train_loss:{},acc:{}'.format(i, j, total_loss, acc))
                    else:
                        print('epoch:{},batch:{},train_loss:{},acc:{}'.format(i, j, total_loss/config.FLAGS.disp_freq, acc))
                    total_loss = 0
            dh.batch_sum = 0
            saver.save(sess, 'ckpt/MutilLSTM.ckpt', global_step=i + 1)
        test_acc = sess.run(accuracy, feed_dict={data: dh.test_x, label: dh.test_y})
        print('model test acc', test_acc)


if __name__ == "__main__":
    LSTM_model()