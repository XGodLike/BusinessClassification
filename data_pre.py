from random import shuffle
import vector_data as vd
import pandas as pd
import numpy as np
data_file = ['F:\\Python\\NLP\\data\\clear_data\\radio.data',
             'F:\\Python\\NLP\\data\\clear_data\\limiteline.data',
             'F:\\Python\\NLP\\data\\clear_data\\movies.data',
             'F:\\Python\\NLP\\data\\clear_data\\music.data',
             'F:\\Python\\NLP\\data\\clear_data\\poi.data',
             'F:\\Python\\NLP\\data\\clear_data\\remind.data',
             'F:\\Python\\NLP\\data\\clear_data\\stock.data',
             'F:\\Python\\NLP\\data\\clear_data\\tel.data',
             'F:\\Python\\NLP\\data\\clear_data\\weather.data']
data_file1 = ['F:\\Python\\NLP\\data\\test.txt']

const_line = 1000 * 1000


def get_allrandomdata():
    c_path = "F:\\Python\\NLP\\data\\radom_20w.csv"
    lines = []
    line = []
    with open(c_path, 'w', encoding='UTF-8') as wf:
        for index, d_file in enumerate(data_file):
            line_count = 0
            with open(d_file, 'r', encoding='UTF-8') as rf:
                words = str(rf.readline()).strip('\n')
                while(words != '' and line_count <= const_line):
                    if(len(words) <= 5):
                        words = str(rf.readline()).strip('\n')
                        continue
                    line.append(words)
                    line.append(index)
                    lines.append(line)
                    line = []
                    words = str(rf.readline()).strip('\n')
                    line_count += 1
                    print(line_count)
    print('start shuffle')
    shuffle(lines)
    df = pd.DataFrame(lines)
    df.to_csv(c_path, mode='a', index=False)


def change_data():
    c_path = "F:\\Python\\NLP\\data\\radom.data"
    lines = []
    with open(c_path, 'w', encoding='UTF-8') as wf:
        for index, d_file in enumerate(data_file1):
            line_count = 0
            with open(d_file, 'rb') as rf:
                line = rf.readline()
                new_line = ''
                words = str(rf.readline())
                words = words.split(' ')
                while(words != [b''] and line_count <= const_line):
                    new_line = words[:len(line):2]
                    new_line = str(new_line) + ' ' + str(index) + '\n'
                    lines.append(new_line)
                    #wf.writelines(new_line)
                    #wf.writelines()
                    #wf.writelines('\n')
                    words = str(rf.readline()).split(' ')
        shuffle(lines)
        for li in lines:
            wf.writelines(li)


def get_random_data(path, random_path):
    lines = []
    with open(path, 'r') as rf:
        line = rf.readline()
        while(line != ''):
            lines.append(line)
            line = rf.readline()
    shuffle(lines)

    with open(random_path, 'w') as wf:
        for li in lines:
            wf.writelines(str(li))


def get_word2vector(path, random_path):
    w2v = vd.WordVector()
    with open(random_path, 'w') as wf:
        with open(path, 'r') as rf:
            line = rf.readline()
            while (line != ''):
                word = line.strip('\n').split(' ')
                word_seq = vd.word_cut(str(word[0]))
                avg_word2vec = str(w2v.get_avg_word2vec(word_seq))
                wf.writelines(avg_word2vec + '  ' + word[1])

                line = rf.readline()


def get_csv(path, random_path):
    w2v = vd.WordVector()
    col = []
    col = ['col' + str(index) for index in range(vd.EMBEDDING_DIM)]
    col.append('label')
    avg_word2vec = []
    line_count = 0
    zero_w2v = True
    with open(path, 'r', encoding='UTF-8') as rf:
        line = rf.readline()
        line_count += 1
        while (line != ''):
            word = line.strip('\n').split(',')
            word_cut = vd.word_cut(str(word[0]))
            zero_w2v, w2v_tmp = w2v.get_avg_word2vec(word_cut)
            # if(zero_w2v == False):
            #     line = rf.readline()
            #     line_count += 1
            #     continue
            list_str = list(w2v_tmp.tolist())
            list_str.append(int(word[1]))
            avg_word2vec.append(list_str)
            line = rf.readline()
            line_count += 1
            if(line_count % 50000 == 0):
                print(line_count)
        df = pd.DataFrame(avg_word2vec, columns=col)
        df.to_csv(random_path, mode='a', index=False)
        avg_word2vec = []


def changetext2csv(path, random_path):
    w2v = vd.WordVector()
    avg_word2vec = []
    line_count = 0
    zero_w2v = True
    with open(path, 'r') as rf:
        line = rf.readline()
        line_count += 1
        while (line != ''):
            print(line_count)
            word = line.strip('\n').split(',')
            # word = line.strip('\n').split(' ')
            # word_cut = vd.word_cut(str(word[0]))
            # zero_w2v, w2v_tmp = w2v.get_avg_word2vec(word_cut)
            # if(zero_w2v == False):
            #     line = rf.readline()
            #     line_count += 1
            #     continue
            list_str = []
            list_str.append(word[0])
            list_str.append(word[1])
            avg_word2vec.append(list_str)
            line = rf.readline()
            line_count += 1
            if(line_count % 1000000 == 0):
                df = pd.DataFrame(avg_word2vec, columns=['text', 'label'])
                df.to_csv(random_path, mode='a', index=False, header=False)
                avg_word2vec = []


def get_keyword_data(path, keyword_path):
    old_tags = []
    new_tags = []
    lines = []
    write_count = 0
    read_count = 0
    with open(keyword_path, 'w', encoding='UTF-8') as wf:
        with open(path, 'r', encoding='UTF-8') as rf:
            line = rf.readline()
            wf.writelines(line)

            line = rf.readline().strip('\n').split(' ')
            words = line[0:len(line):2]
            old_tags = line[1:len(line):2]

            while words != [''] and write_count <= const_line:
                if len(words) <= 5:
                    line = rf.readline().strip('\n').split(' ')
                    words = line[0:len(line):2]
                    old_tags = line[1:len(line):2]
                    continue
                line = ''.join(words)
                cut_words = vd.word_cut(line).split(' ')
                tag_tmp = ''
                TrueBreak = False
                for i in range(len(cut_words)):
                    if TrueBreak:
                        break
                    for j in range(i, len(words)):
                        if cut_words[i] == words[j]:
                            new_tags.append(old_tags[j])
                            tag_tmp = ''
                            break
                        elif words[j] in cut_words[i]:
                            if tag_tmp != '' and tag_tmp != old_tags[j]:
                                TrueBreak = True
                                break
                            elif j != len(words)-1 and words[j+1] in cut_words[i]:
                                tag_tmp = old_tags[j]
                            elif tag_tmp == '':
                                TrueBreak = True
                                break
                            else:
                                new_tags.append(tag_tmp)
                                tag_tmp = ''
                                break

                if not TrueBreak and len(new_tags) == len(cut_words):
                    new_line = ''
                    for i in range(len(new_tags)):
                        new_line += cut_words[i].strip(' ').strip('\n')
                        new_line += ' '
                        new_line += new_tags[i].strip(' ').strip('\n')
                        if i == len(new_tags)-1:
                            new_line += '\n'
                        else:
                            new_line += ' '
                    lines.append(new_line)
                    write_count += 1
                    print('write_count = ', write_count)
                line = rf.readline().strip('\n').split(' ')
                words = line[0:len(line):2]
                old_tags = line[1:len(line):2]
                read_count += 1
                print('read_count = ', read_count)
                new_tags = []
                cut_words = []
        print('start shuffle')
        shuffle(lines)
        for i in range(len(lines)):
            wf.writelines(lines[i])
        print('write done')




if __name__ == "__main__":
    #生成随机数据。每个业务包含20W条
    #get_allrandomdata()
    #change_data()
    #get_random_data('data.data', 'random.data')
    #get_word2vector('random.data', 'classfication.data')
    #changetext2csv('data.csv', 'test.csv')
    #get_csv('F:\\Python\\NLP\\data\\radom_10w.csv', 'w2v_60.csv')#
    #生成导航关键词数据，每个业务
    get_keyword_data("F:\\Python\\NLP\\data\\old_data\\limiteline_train_170822_rnd.txt", "F:\\Python\\NLP\\data\\limiteline.txt")