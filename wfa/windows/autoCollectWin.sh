#!/bin/bash

# nvcc ../../receiver.cu -o ./a.exe

# logfiles
LOGFILE="./openweb_chrome.log"

# paramteters
LOOPTIME=250

TIME_START=3
TIME_HOMEPAGE=2
TIME_TARGETPAGE=10

source weblabels.sh

# initialize
time_next=$(date +%s)
time_curr=$(date +%s)

nircmd.exe win activate ititle "Google Chrome"
nircmd.exe win max ititle "Google Chrome"
cscript.exe //B sdkey.vbs "%{HOME}"

echo "------------------ start --------------------" >> $LOGFILE
echo start at $(date +%D%r) >> $LOGFILE
time_next=$(($time_next + $TIME_START))

for url in "${URL_ARRAY[@]}"; do
  mkdir -p "chromeData/$url" 
  for ((i = 0; i < $LOOPTIME; i++)); do
    # open target websites 
    # while [ $time_curr -lt $time_next ]; do
    #   time_curr=$(date +%s)
    # done
    echo open $url at $(date +%D%r) >> $LOGFILE
    cscript.exe //B sdkey.vbs "^l"
    sleep 0.5
    cscript.exe //B sdkey.vbs $url
    sleep 0.5
    ./a.exe 7 > chromeData/$url/${i}.txt "chromeData/$url/" &
    sleep 0.5
    cscript.exe //B sdkey.vbs "{ENTER}"
    # time_next=$(($time_next + $TIME_TARGETPAGE))
    sleep $TIME_TARGETPAGE	
    # return to home page to wait
    # while [ $time_curr -lt $time_next ]; do
    #   time_curr=$(date +%s)
    # done
    echo return to homepage at $(date +%D%r) >> $LOGFILE
    cscript.exe //B sdkey.vbs "%{HOME}"
    sleep 0.5
    # time_next=$(($time_next + $TIME_HOMEPAGE))
  done
done

# enter blank page 
cscript.exe //B sdkey.vbs "^l"
cscript.exe //B sdkey.vbs "about:blank"
cscript.exe //B sdkey.vbs "{ENTER}"

echo "------------------- end ---------------------" >> $LOGFILE
