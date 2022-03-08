#!/usr/bin/python3

import sys
import os
import keras
import numpy as np

def readucr(filename):
  data = np.loadtxt(filename, delimiter = ',')
  Y = data[:,0]
  X = data[:,1:]
  return X, Y

def normalizelist(x):
  z = []
  for y in x:
    ymax = max(y)
    ymin = min(y)
    w = []
    for i in range(len(y)):
      w.append((y[i] - ymin) / (ymax - ymin))
    z.append(w)
  return np.array(z)

# load model
model = keras.models.load_model(sys.argv[1], compile = True)

x_test, y_test = readucr(sys.argv[2])

x_test = normalizelist(x_test)
#for x in x_test:
#  print(x)
#  input()
x_test = x_test.reshape(x_test.shape + (1,1,))

y_test = y_test - 1

save_flag = False
if len(sys.argv) > 3:
  try:
    file_result = open(sys.argv[3], "w")
    save_flag = True
  except:
    print("Cannot open file " + sys.argv[3])

pred = model.predict(x_test)
j = 0

for i in range(len(pred)):
  res = pred[i]
  #if np.argmax(res) == 0:
  #  print([np.max(res), np.argmax(res), y_test[i]])
  if np.argmax(res) != int(y_test[i]):
    print([np.max(res), np.argmax(res), int(y_test[i])])
    if save_flag:
      print([np.max(res), np.argmax(res), int(y_test[i])], file=file_result)

    #print(["true:", int(y_test[i]) + 1, "predicted:", np.argmax(res) + 1])
    j = j + 1
print(j)

if save_flag:
  file_result.close()

#if np.max(res) > 0.91:
#if np.max(res) > 0.01:

