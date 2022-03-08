#!/bin/bash

nvcc ../../receiver.cu -o ./wfa.out

# websites top 100
declare -a websites=( 
  # 1-10
  "Adobe.com"
  "Aliexpress.com" 
  "Alipay.com" 
  "Allrecipes.com"
  "Amazon.com" 
  "Aol.com"
  "Apartments.com"
  "Apple.com"
  "Att.com"
  "Baidu.com"

  # 11-20
  "Bankofamerica.com"
  "Bbc.com"
  "Bestbuy.com"
  "Bing.com" 
  "Blogger.com" 
  "Britannica.com"
  "Businessinsider.com"
  "Ca.gov"
  "Capitalone.com"
  "Cdc.gov"
  
  # 21-30
  "Chase.com"
  "Cheatsheet.com"
  "Cnn.com"
  "Costco.com"
  "Craigslist.org"
  "Dailymail.co.uk"
  "Duckduckgo.com"
  "Ebay.com"
  "Espn.com"
  "Etsy.com"

  # 31-40
  "Expedia.com"
  "Facebook.com" 
  "Fandom.com"
  "Fedex.com"
  "Fidelity.com"
  "Foxnews.com"
  "Gamepedia.com"
  "Github.com"
  "Glassdoor.com"
  "Google.com" 

  # 41-50
  "Healthline.com"
  "Homedepot.com"
  "Hulu.com"
  "Ign.com"
  "Imdb.com"
  "Imgur.com"
  "Indeed.com"
  "Instagram.com" 
  "Intuit.com"
  "Irs.gov"

  # 51-60
  "Linkedin.com"
  "Live.com" 
  "Lowes.com"
  "Mayoclinic.org"
  "Merriam-webster.com"
  "Microsoft.com" 
  "Msn.com"
  "Nbcnews.com"
  "Netflix.com"
  "Nfl.com"

  # 61-70
  "Nih.gov"
  "Npr.org"
  "Nypost.com"
  "Nytimes.com"
  "Office.com" 
  "Paypal.com"
  "Pinterest.com"
  "Quizlet.com"
  "Quora.com"
  "Realtor.com"

  # 71-80
  "Reddit.com" 
  "Rottentomatoes.com"
  "Shopify.com" 
  "Speedtest.net"
  "Spotify.com"
  "Stackoverflow.com"
  "T-mobile.com"
  "Target.com"
  "Tripadvisor.com"
  "Twitch.tv" 

  # 81-90
  "Twitter.com"
  "Ups.com"
  "Usa.gov"
  "Usps.com"
  "Vk.com" 
  "Walmart.com"
  "Washingtonpost.com"
  "Weather.com"
  "Weather.gov"
  "Webmd.com"

  # 91-100
  "Weibo.com" 
  "Wellsfargo.com"
  "Wikipedia.org" 
  "Xfinity.com"
  "Yahoo.com" 
  "Yandex.com"
  "Yelp.com"
  "Youtube.com" 
  "Zillow.com"
  "Zoom.us" 
)

NO1=250
for web in "${websites[@]}"; do
  mkdir -p "chromeData/${web}" 
  for i in $(seq 1 $NO1); do 
    wmctrl -a "Google Chrome"
    xdotool key Alt+Home
    sleep 0.5
    xdotool key Ctrl+l
    sleep 0.5
    xdotool type "www.${web}"
    sleep 1
    ./wfa.out 7 > chromeData/${web}/${i}.txt "chromeData/${web}/${i}" & 
    sleep 1
    xdotool key Return
    sleep 10
  done
done
