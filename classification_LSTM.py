import tensorflow as tf
import numpy as np
import data_help
import config
import vector_data as vd


class LSTMModel:
    def __init__(self, num_classes, isTraining=True):
        self.__dropout_prob = config.FLAGS.keep_prob
        self.__hidden_layers = config.FLAGS.hidden_layer_num
        self.__hidden_units = config.FLAGS.lstm_hidden_units
        self.__num_step = config.FLAGS.max_len
        self.__IsTraining = isTraining
        self.__num_class = num_classes
        self._initial_state = None
        self.cell = None
        self.outputs = []
        self.X = tf.placeholder(tf.float32, [None, self.__num_step, config.FLAGS.embed_dim], name='embedded_word')
        self.Y = tf.placeholder(tf.float32, [None, self.__num_class], name='target')
        self.mask_x = tf.placeholder(tf.float32, [self.__num_step, None], name="mask_x")

        self.predition = 0.0
        self.loss = 0.0
        self.acc = 0.0
        self.train_op = None

    def __lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(config.FLAGS.lstm_hidden_units)

    def __drorpout_cell(self):
        if self.__IsTraining and config.FLAGS.keep_prob < 1:
            return tf.contrib.rnn.DropoutWrapper(self.__lstm_cell(),  output_keep_prob=config.FLAGS.keep_prob)
        else:
            return self.__lstm_cell()

    def net_cell(self):
        self.cell = tf.contrib.rnn.MultiRNNCell([self.__drorpout_cell() for _ in range(config.FLAGS.hidden_layer_num)], state_is_tuple=True)
        self._initial_state = self.cell.zero_state(config.FLAGS.train_batch_size, tf.float32)

    def get_in(self):
        if self.__IsTraining and config.FLAGS.keep_prob < 1:
            inputs = tf.nn.dropout(self.X, config.FLAGS.keep_prob)
        else:
            inputs = self.X
        return inputs

    def get_out(self):
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.__num_step):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(self.X[:, time_step, :], state) #inputs有三个维度，第一个代表batch中的第几个样本，2是样本中第几个单词，3是单词的向量表达的维度
                self.outputs.append(cell_output)
        self.outputs = self.outputs * self.mask_x[:, :, None]

    def mean_pooling_layer(self):
        with tf.name_scope('mean_pooling_layer'):
            self.outputs = tf.reduce_mean(self.outputs, 0)/(tf.reduce_sum(self.mask_x, 0)[:, None])

    def w_plus_b(self):
        with tf.name_scope('softmax_layer_and_output'):
            w = tf.get_variable('w', [self.__hidden_units, self.__num_class], dtype=tf.float32)
            b = tf.get_variable('b', [self.__num_class], dtype=tf.float32)
            #self.scores = tf.matmul(self.outputs, w) + b
            self.predition = tf.nn.xw_plus_b(self.outputs, w, b, name='prediction')

    def computloss(self):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.predition)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')

    def optimization(self):
        with tf.name_scope('Optimization'):
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def accuracy(self):
        correct_prediction = tf.equal(tf.arg_max(self.predition, 0), tf.arg_max(self.Y, 0))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        self.net_cell()
        self.get_in()
        self.get_out()
        self.mean_pooling_layer()
        self.w_plus_b()
        self.computloss()
        self.optimization()

def load_data(path):
    # 数据输入
    dh = data_help.DataHelper(path)
    print('Data loaded...')
    dh.split_train_validation_test()
    print('Data splited...')



def lstm_model(path):
    dh = data_help.DataHelper(path)
    print('Data loaded...')
    dh.split_train_validation_test()
    print('Data splited...')

    lstm = LSTMModel(9)
    lstm.train()

    print('~~~~~~~~~~~开始执行计算图~~~~~~~~~~~~~~')
    saver = tf.train.Saver(max_to_keep=3)# 保存训练的中间模型
    n_batches = int(dh.train_length / config.FLAGS.train_batch_size)

    with tf.Session() as sess:
        feed_dict = {}
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logdir=config.FLAGS.log_dir + '/train', graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=config.FLAGS.log_dir + '/test')
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        for i in range(0, config.FLAGS.epoch):
            for j in range(0, n_batches):
                X_batch, Y_batch = dh.next_batch(config.FLAGS.train_batch_size)
                _, loss_batch, summary = sess.run([lstm.train_op, lstm.loss, merged], feed_dict={lstm.X: X_batch, lstm.Y: Y_batch})
                train_writer.add_summary(summary)
                total_loss += loss_batch
                if j % config.FLAGS.disp_freq == 0:
                    val_acc, summary = sess.run([lstm.acc, merged], feed_dict={lstm.X: dh.val_x, lstm.Y: dh.val_y})
                    if j == 0:
                        print('epoch :{}, step: {}, train_loss: {}, val_acc: {}'.format(i, j, total_loss, val_acc))
                    else:
                        print('epoch :{}, step: {}, train_loss: {}, val_acc: {}'.format(i, j, total_loss / config.FLAGS.disp_freq, val_acc))
                    test_writer.add_summary(summary, j)
                    total_loss = 0
            dh.batch_sum = 0
            saver.save(sess, 'ckpt/classfication_softmax.ckpt', global_step=i + 1)
        test_acc = sess.run(lstm.acc, feed_dict={lstm.X: dh.test_x, lstm.Y: dh.test_y})
        print('test accuracy: {}'.format(test_acc))
        train_writer.close()
        test_writer.close()

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

    with tf.name_scope('Infrence'):
        weight = tf.Variable(tf.random_normal(shape=[60, 9]), name='weight')
        bias = tf.Variable(initial_value=tf.zeros(shape=[9]), name='bias')
        Wx_plus_b = tf.matmul(x, weight) + bias
        y_prediction = tf.nn.softmax(Wx_plus_b)
    #layer1 = add_lay(x, 60, 30, activate_function=tf.nn.tanh)
    #layer2 = add_lay(layer1, 30, 30, activate_function=tf.nn.tanh)
    #y_prediction = add_layer(x, 60, 9, activate_function=tf.nn.softmax)


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
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logdir=config.FLAGS.log_dir + '/train', graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=config.FLAGS.log_dir + '/test')

        sess.run(tf.global_variables_initializer())
        total_loss = 0
        for i in range(0, config.FLAGS.epoch):
            for j in range(0, n_batches):
                X_batch, Y_batch = dh.next_batch(config.FLAGS.train_batch_size)
                _, loss_batch, summary = sess.run([train_op, loss, merged], feed_dict={x: X_batch, y: Y_batch})
                train_writer.add_summary(summary, j)
                total_loss += loss_batch
                if j % config.FLAGS.disp_freq == 0:
                    val_acc, summary = sess.run([accuracy, merged], feed_dict={x: dh.val_x, y: dh.val_y})
                    test_writer.add_summary(summary, j)
                    if j == 0:
                        print('epoch :{}, step: {}, train_loss: {}, val_acc: {}'.format(i, j, total_loss, val_acc))
                    else:
                        print('epoch :{}, step: {}, train_loss: {}, val_acc: {}'.format(i, j, total_loss / config.FLAGS.disp_freq, val_acc))
                    total_loss = 0
            dh.batch_sum = 0
            saver.save(sess, 'ckpt/classfication_softmax.ckpt', global_step=i + 1)
        test_acc = sess.run(accuracy, feed_dict={x: dh.test_x, y: dh.test_y})
        print('test accuracy: {}'.format(test_acc))
        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    print('Start...')
    softmax_model()
    #load_data('F:\\Python\\NLP\\data\\radom_10w.csv')
    #lstm_model('F:\\Python\\NLP\\data\\radom_10w.csv')
