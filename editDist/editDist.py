# Python3 code to demonstrate working of
# Converting String to binary
# Using join() + bytearray() + format()

import sys
#import nltk
import editdistance

print("Calculating Levenshtein Distance...")

receivedFile = "binRec.txt"
if len(sys.argv) > 1:
  receivedFile = sys.argv[1]
derivedStr = open(receivedFile).read()
print("\nreceivedFile: " + receivedFile)
# print("derivedStr: \n" + derivedStr)
print("derivedStr len: " + str(len(derivedStr)))

baseFile = "binBase.txt"
if len(sys.argv) > 2:
  baseFile = sys.argv[2]
baseStr = open(baseFile).read()
print("\nbaseFile: " + baseFile)
# print("Base: \n" + base)
print("baseStr len: " + str(len(baseStr)))

print("\n...calculating...\n")

# editDist = nltk.edit_distance(derivedStr, base)

editDist = editdistance.eval(derivedStr, baseStr)

print("Edit distance: " + str(editDist) + "\n")

print("editDist/baseLen: " + str(editDist/len(baseStr) * 100) + " %\n")