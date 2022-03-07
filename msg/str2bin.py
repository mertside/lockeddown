# Python3 code to demonstrate working of
# Converting String to binary
# Using join() + bytearray() + format()

import sys

# initializing string 
test_str = "MertSIDE"
if len(sys.argv) > 1:
  test_str = open(sys.argv[1]).read()

if len(sys.argv) > 2:
  bin_msg = open(sys.argv[2], "w")

# printing original string 
print("The original string is : \n" + str(test_str))
  
# using join() + bytearray() + format()
# Converting String to binary
res = ''.join(format(i, '08b') for i in bytearray(test_str, encoding ='utf-8'))
  
# printing result 
print("The string after binary conversion : \n" + str(res))

if len(sys.argv) > 2:
  for x in res:
    bin_msg.write(x)
    #bin_str.write('\n')
  bin_msg.close()
