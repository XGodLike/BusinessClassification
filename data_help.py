from sklearn.model_selection import train_test_split
import numpy as np
import vector_data as vd
import pandas as pd
import config
from random import shuffle

LSTM = True
KEY_WORD = False
FIRST = True

max_len = vd.max_len
EMBEDDING_DIM = vd.EMBEDDING_DIM

classfication_data_file = 'F:\\Python\\NLP\\input.npy'
classfication_target_file = 'F:\\Python\\NLP\\target.npy'

kw_data_file = 'F:\\Python\\NLP\\keyword\\kw_input_50w.npy'
kw_target_file = 'F:\\Python\\NLP\\keyword\\kw_target_50w.npy'

class DataHelper:
    def __init__(self, path):
        self.__path = path
        self.__data = []
        self.__target = []
        self.train_length = 0
        self.length = 0
        if LSTM:
            if KEY_WORD:
                self.tag_count = 0
                self.__get_KeyWord_data()
            else:
                self.__get_LSTM_data()
        else:
            self.__get_data()
        self.mask_x = []
        self.train_x = []
        self.train_y = []
        self.val_x = []
        self.val_y = []
        self.test_x = []
        self.test_y = []
        self.batch_sum = 0

    def __generate_mask(self, data):
        max_len = config.FLAGS.max_len
        set_x = data
        self.mask_x = np.zeros([max_len, len(set_x)])
        for i, x in enumerate(set_x):
            x_list = x.split(' ')
            if len(x_list) < max_len:
                self.mask_x[0:len(x_list), i] = 1
            else:
                self.mask_x[0:, i] = 1

    def __get_KeyWord_data(self):
        if FIRST:
            dict = {}
            w2v = vd.WordVector()
            one_tag = []
            count = 0
            tag_count = 0
            with open(self.__path, 'r', encoding='UTF-8') as rf:
                line = rf.readline().strip('\n').split(' ')
                self.tag_count = len(line)
                for i in range(self.tag_count):
                    label = np.zeros(self.tag_count)
                    label[i] = 1
                    dict[line[i].lower()] = label
                line = rf.readline().strip('\n').split(' ')
                word = line[0:len(line):2]
                tag = line[1:len(line):2]

                while line != ['']:
                    count += 1
                    if count % 100000 == 0:
                        print(count)
                    for j in range(max_len):
                        if j < len(tag):
                            one_tag.append(dict[tag[j].lower()])
                        else:
                            one_tag.append(np.zeros(self.tag_count))
                    self.__data.append(w2v.get_each_word2vec(' '.join(word)))
                    self.__target.append(one_tag)
                    one_tag = []
                    line = rf.readline().strip('\n').split(' ')
                    word = line[0:len(line):2]
                    tag = line[1:len(line):2]
            self.length = len(self.__data)
            print(self.length)
            np.save(kw_data_file, self.__data)
            np.save(kw_target_file, self.__target)
        else:
            with open(self.__path, 'r', encoding='UTF-8') as rf:
                line = rf.readline().strip('\n').split(' ')
                self.tag_count = len(line)
            self.__data = np.load(kw_data_file)
            self.__target = np.load(kw_target_file)
            self.length = len(self.__data)

    def __get_LSTM_data(self):
        if FIRST:
            reader = pd.read_csv(self.__path, iterator=True, encoding='UTF-8')
            loop = True
            chunkSize = 500000
            chunks = []
            while loop:
                try:
                    chunk = reader.get_chunk(chunkSize)
                    chunks.append(chunk)
                except StopIteration:
                    loop = False
                    print("Iteration is stopped.")
            df = pd.concat(chunks, ignore_index=True)
            self.length = df.shape[0]
            w2v = vd.WordVector()
            for index in range(df.shape[0]):
                if index % 100000 == 0:
                    print(index)
                label = np.zeros(9)
                words = df.iloc[index].tolist()
                wc = vd.word_cut(words[0])
                #self.mask_x.append(self.__generate_mask(wc))
                self.__data.append(w2v.get_each_word2vec(wc))
                label[int(words[-1])] = 1
                self.__target.append(label)
            #np.save(classfication_data_file, self.__data)
            #np.save(classfication_target_file, self.__target)
        else:
            self.__data = np.load(classfication_data_file)
            self.__target = np.load(classfication_target_file)
            self.length = len(self.__data)

    def __get_data(self):
        reader = pd.read_csv(self.__path, iterator=True, encoding='UTF-8')
        loop = True
        chunkSize = 500000
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        df = pd.concat(chunks, ignore_index=True)
        ##################################################################
        self.length = df.shape[0]
        for index in range(df.shape[0]):
            label = np.zeros(9)
            words = df.iloc[index].tolist()
            self.__data.append(np.array(words[:-1]))
            label[int(words[-1])] = 1
            self.__target.append(label)

    def get_data(self):
        data = []
        target = []
        w2v = vd.WordVector()
        with open(self.__path, 'r') as rf:
            lines = rf.readline()
            while(lines != ''):
                print(lines)
                y = np.zeros([9, 1])
                sp_line = str(lines).split(' ')
                word_cut = vd.word_cut(sp_line[0])
                _, word2vec = w2v.get_avg_word2vec(word_cut)
                if(_ == False):
                    lines = rf.readline()
                    continue

                data.append(sp_line[0])
                y[int(sp_line[1])] = 1
                target.append(y)
                lines = rf.readline()
        assert (len(data) == len(target))
        self.length = len(data)

    def split_train_test(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.__data, self.__target, test_size=0.05, random_state=33)

#由于文本本身已经随机排序过，这里采用按顺序的方法split数据集，后面打算再次随机取
    def split_train_validation_test(self, val_percent=0.04, test_percent=0.01):
        train_size = int(self.length * (1 - val_percent - test_percent))
        val_size = int(self.length * val_percent)
        self.train_x = np.asarray(self.__data[:train_size], dtype=np.float32)
        self.train_y = np.asarray(self.__target[:train_size], dtype=np.float32)
        self.val_x = np.asarray(self.__data[train_size:train_size + val_size], dtype=np.float32)
        self.val_y = np.asarray(self.__target[train_size:train_size + val_size], dtype=np.float32)
        self.test_x = np.asarray(self.__data[train_size + val_size:self.length], dtype=np.float32)
        self.test_y = np.asarray(self.__target[train_size + val_size:self.length], dtype=np.float32)
        self.train_length = self.train_x.shape[0]
        return (self.train_x, self.train_y), (self.val_x, self.val_y), (self.test_x, self.test_y)

    def shuffle_batch(self):
        shuffle(self.train_x)
        shuffle(self.train_y)

    def next_batch(self, batch_size=100):
        self.batch_sum += batch_size
        if(self.batch_sum <= self.train_length):
            X_train = self.train_x[self.batch_sum-batch_size:self.batch_sum]
            Y_train = self.train_y[self.batch_sum-batch_size:self.batch_sum]
            return X_train, Y_train
        else:
            X_train = self.train_x[self.batch_sum-batch_size:self.train_length]
            Y_train = self.train_y[self.batch_sum-batch_size:self.train_length]
            return X_train, Y_train

    def next_batch_shuffle(self, batch_size=100):
        X_train = shuffle(self.train_x[0:batch_size])
        Y_train = shuffle(self.train_y[0:batch_size])
        return X_train, Y_train



if __name__ == '__main__':
    dh = DataHelper('F:\\Python\\NLP\\data\\poi_36w.txt')
    dh.split_train_validation_test()
    dh.next_batch(5)
    # print(dh.train_x[0])
    # print(dh.train_y[0])
    # #print(dh.val_x[0])
    # #print(dh.val_y[0])
    # print(dh.test_x[0])
    # print(dh.test_y[0])

