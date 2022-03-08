import matplotlib.pyplot as plt
import csv

yMin = 7500000   #9000000
yMax = 15000000  #20000000

x = []
y = []

websites = [ 
  # 1-10
  "Google.com",
  "Youtube.com", 
  "Amazon.com",
  "Wikipedia.org", 
  "Facebook.com",
  "Reddit.com", 
  "Yahoo.com",
  "Netflix.com",
  "Ebay.com",
  "Twitter.com"
]
 
workspace = [
  # Ubuntu:  
  "/home/mside/mertWork/gitProjects/webprint",
  # macOS:    
  "/Users/MertSide/Developer/GitProjects/webPrint"
]

directory = workspace[1]
suffix = "/"                                        # adjust
websiteName = websites[9]
folder = "/chromeData" + suffix + websiteName + "/"   # adjust
filename = "2"                                        # adjust
fileExt = ".txt"
fullPath = directory + folder + filename + fileExt

with open(fullPath,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))

#plt.plot(x,y, label='Loaded from file!')
plt.plot(x,y)
plt.xlabel('datapoint')
# plt.ylim(yMin,yMax)
plt.ylabel('rdtsc (cycles)')
plt.title(folder + filename + fileExt + '\n')
# plt.legend()
# plt.show()
plt.savefig(directory+"/chromePlot"+suffix+websiteName+"-fig-"+filename+".png",dpi=300)