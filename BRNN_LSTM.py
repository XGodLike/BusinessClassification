import pandas as pd
import numpy as np
import tensorflow as tf

aaa= np.asarray([[[1,2,3],[2,3,4]]])
print(aaa[0])

firstSentence = np.zeros((10), dtype='int32')
print(firstSentence)

print(int(1.599))
t = np.zeros([1,9])
print(t)
t[0][3] = 1
print(t)

a = [[1],[4],[9]]
print(tf.reduce_sum(a))
print(tf.reduce_mean(a))
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(a)))
    print(sess.run(tf.reduce_mean(a)))
    print(sess.run(tf.argmax(a, 0)))

a=([3.234,34,3.777,6.33])
# print np.array(a)

df = pd.DataFrame([[1,2,'c'],['d',3 ,'f']], columns=['1', '2', '3'])
df.to_csv('1.csv', index=False, header=False)
rf = pd.read_csv('1.csv', header=None)
a = []
b = []
for index in range(rf.shape[0]):
    list_da = rf.iloc[index].tolist()
    a.append(list_da[:-1])
    b.append(list_da[-1])
print(list_da, a, b)
