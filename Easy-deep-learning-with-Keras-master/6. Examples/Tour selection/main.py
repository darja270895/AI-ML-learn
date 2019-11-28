from numpy import loadtxt
import pandas

import pyexcel as pe
import numpy

import xlrd #for excel reading

from keras.models import Sequential
from keras.layers import Dense





dataset = pe.get_array(file_name = 'dataset.xls')


# dataset = pandas.read_excel('dataset.xls', header=None)
# arr = dataset._get
# print(arr)

new_list = []
for i in dataset:
    temp = []
    for j in i:
        byt = j.encode('utf-8')
        to_int = int.from_bytes(byt, "big")
        temp.append(to_int)
    new_list.append(temp)

arr = numpy.asarray(new_list)

x = arr[:, 0:11]
y = arr[:, 11]


import struct






model = Sequential()
model.add(Dense(11, input_dim=11, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=150, batch_size=10)


_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))


# make probability predictions with the model
predictions = model.predict(x)
# # round predictions
rounded = [round(x[0]) for x in predictions]

# # make class predictions with the model
# # predictions = model.predict_classes(X)

for i in range(10):
	print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))