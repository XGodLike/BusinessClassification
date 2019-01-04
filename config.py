import tensorflow as tf

# 定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数
FLAGS = tf.app.flags.FLAGS

# 设置训练相关参数
tf.app.flags.DEFINE_integer("lstm_hidden_units", 10, "hidden neural lstm cell numbers")
tf.app.flags.DEFINE_integer("hidden_layer_num", 2, "hidden layer number")
tf.app.flags.DEFINE_float("keep_prob", 0.75, "dropout probility")
tf.app.flags.DEFINE_integer("embed_dim", 60, "embedding word dimision")
tf.app.flags.DEFINE_integer("max_len", 10, "the most long sequence length")
tf.app.flags.DEFINE_integer("epoch", 20, "Iterations to train [1e4]")
#tf.app.flags.DEFINE_integer("iteration", 1000, "Iterations to train [1e4]")
tf.app.flags.DEFINE_integer("disp_freq", 1000, "Display the current results every display_freq iterations [1e2]")
tf.app.flags.DEFINE_integer("train_batch_size", 100, "The size of batch images [128]")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate of for adam [0.01]")
tf.app.flags.DEFINE_string("log_dir", "logs", "Directory of logs.")
