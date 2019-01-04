# -*- coding:UTF-8 -*-
import numpy as np
import gensim
import jieba
from gensim.models import Word2Vec, KeyedVectors

#wv_path = 'word_vector.bin'
wv_path = 'Word60.model'
EMBEDDING_DIM = 60

max_len = 10
jieba.load_userdict('user.dict')
class WordVector:
        def __init__(self, path=wv_path):
            #self.__model = KeyedVectors.load_word2vec_format(path, binary=True) #C版本加载模型，不能用于再训练
            self.__model = Word2Vec.load(path)#加载模型，可以用于再训练
            print('Loaded WordVectorModel...')
            self.__embedding_dim = EMBEDDING_DIM

        def get_each_word2vec(self, seq_list):
            word2vec = []
            words = seq_list.strip('\n').split(' ')
            word_count = 0
            for tmp in words:
                if word_count == max_len:
                    break
                word2vec.append(np.asarray(self.__get_word2vec(tmp)))
                word_count += 1

            while word_count < max_len:
                word2vec.append(np.zeros(60))
                word_count += 1

            return word2vec


        def get_avg_word2vec(self, seq_list):
            #assert (type(seq_list) == str)
            words = seq_list.strip('\n').split(' ')
            word = np.zeros(EMBEDDING_DIM)
            for tmp in words:
                word += self.__get_word2vec(tmp)
            avg = word/len(words)
            if(avg.max() == 0.0):
                return False, avg
            else:
                return True, avg

        def __get_word2vec(self, word):
            if(word in self.__model):
                return self.__model[word]
            else:
                return np.zeros(EMBEDDING_DIM)


def word_cut(word):
        seg_list = jieba.cut(word, cut_all=False)
        return ' '.join(seg_list)

if __name__ == "__main__":
    w2v = WordVector(wv_path)
    jieba.load_userdict('user.dict')
    print(word_cut('二零二八一月二十号限行C还是九'))
    print(w2v.get_avg_word2vec(u'春雷'))