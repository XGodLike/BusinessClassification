

def clear_data(In_path, Out_path):
    with open(Out_path, 'w', encoding='UTF-8') as wf:
        with open(In_path, 'r', encoding='UTF-8') as rf:
            line = rf.readline().strip('\n')
            while(line != ''):
                if(len(line) <= 13):
                    line = rf.readline().strip('\n')
                    continue
                else:
                    Writenable = False
                    for i in range(len(line)):
                        if(line[i] <= 'Z' and line[i] >= 'A'):
                            continue
                        else:
                            Writenable = True
                            break
                    if(Writenable == True):
                        wf.writelines(line)
                        wf.writelines('\n')
                    line = rf.readline().strip('\n')





if __name__ == '__main__':
    clear_data('F:\\Python\\NLP\\data\\limiteline_train_170822_rnd.data', 'F:\\Python\\NLP\\data\\clear_data\\limiteline.data')