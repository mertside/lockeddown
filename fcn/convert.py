#!/usr/bin/python3

import sys
import os
import numpy as np
import random
import re

path0 = sys.argv[1]
dirs = next(os.walk(path0))[1]
# dirs.sort() # not neccessary for this file org

i = 0             # website number
trainseq = list()
testseq = list()
numberOfTest = int(sys.argv[3])

for d0 in dirs: 
  path1 = os.path.join(path0, d0)
  print(path1)
  subs = [x for x in os.listdir(path1) if os.path.isdir(os.path.join(path1, x))]
  subs.sort()
  # print(subs)
  
  # process each website
  for d1 in subs:
    i = i + 1     # website number
    j = 0         # file number for a given website

    path2 = os.path.join(path1, d1)
    print("\t" + path2)
    
    files = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    if len(files) == 0:
      continue
    
    # process each file
    seq = list()
    for f in files:
      path3 = os.path.join(path2, f)
      
      f_in = open(path3)    
      cleanStr = ''
      # get rid of the number before comma
      # cause my traces are formatted as "number,data"
      for l_in in f_in:
        tmpStr = ','.join(l_in.split(',')[1:])
        cleanStr = cleanStr + tmpStr
      
      lines = cleanStr.split('\n')

      trace = str(i)
      k = 0
      for l in lines:
        k = k + 1
        if k <= (len(lines)-1):
          trace = trace + "," 
        trace = trace + str(l)
      # print(k)
      # print(len(lines))
      seq.append(trace)
    random.shuffle(seq)

    for s in seq:
      j = j + 1   # file number for a given website
      if j <= numberOfTest :
        testseq.append(s)
      else:
        trainseq.append(s)
    # print(j)
    # i = i + 1     # website number

random.shuffle(trainseq)

trainfile = open(sys.argv[2] + "_TRAIN", "w")
for x in trainseq:
  trainfile.write(x)
  trainfile.write('\n')
trainfile.close()

testfile = open(sys.argv[2] + "_TEST", "w")
for x in testseq:
  testfile.write(x)
  testfile.write('\n')
testfile.close()
