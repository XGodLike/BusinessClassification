# -*- coding: utf-8 -*-
"""
Spyder Editor
作者ZL
This is a temporary script file.
"""

from gensim.models import Word2Vec

#文本文件必须是utf-8无bom格式
mod = Word2Vec.load(r'Word60.model')
fout = open(r"字词相似度.txt", 'w')

showWord=['导航',
'雍和宫',
'的',
'我',
'你',
'他',
'个',
'1',
'完成',
'吃',
'苹果',
'香蕉',
'词汇',
'物理',
'地球',
'黑死病',
'瘟疫',
'',]

for word in showWord:
	if word in mod:
		print(mod[word])
		sim = mod.most_similar(word)
		fout.write(word +'\n')
		for ww in sim:
			fout.write('\t\t\t' + ww[0] + '\t\t'  + str(ww[1])+'\n')
		fout.write('\n')
	else:
		fout.write(word + '\t\t\t——不在词汇表里'+'\n\n')

fout.close()  
