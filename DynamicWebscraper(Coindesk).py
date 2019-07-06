########################################################################################################################
########################################################################################################################
###############         Scraping Coindesk for                ###########################################################
###############         Blockchain news links                ###########################################################
###############              (Coindesk)                      ###########################################################
########################################################################################################################
########################################################################################################################


import os
import pandas as pd
import time
from bs4 import BeautifulSoup as soup
from selenium import webdriver       #an API that provides support for modern dynamic web page testing problems
from selenium.webdriver.common.keys import Keys
import re

url = "https://www.coindesk.com/category/business-news"

#If not by default set path for chromedriver
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(chrome_options=options,
                          executable_path=r"/Users/alexd/Desktop/Masterarbeit/Python/LSFM/chromedriver")
driver.get(url)

#Check for latest version of chromedriver and support for chrome!


html = driver.page_source.encode('utf-8')
page_num = 0
while page_num < 1500:   #Number of times load_more button is clicked
    article = driver.find_element_by_xpath('//*[@id="load-more-stories"]/button')
    #Execute_script method needed as load more button is not visible at all times.
    driver.execute_script('arguments[0].click();', article)
    page_num += 1
    print("getting page number "+str(page_num))
    time.sleep(1)


#Links get stored in csv file to scrape the text later on:
html = driver.page_source
mysoup = soup(html, "html.parser")

filename = "bclinks.csv"
f = open(filename, 'w')
headers = "linkname\n"
f.write(headers)
link_num = 1

for link in mysoup.findAll("a", href = re.compile("^(https://www.coindesk.com/)")):
 if 'href' in link.attrs:
  linkname = link.attrs['href']
  print("This is link "+ str(link_num))
  f.write(linkname + "\n")
  link_num += 1

f.close()
driver.close()

##Additional cleaning:
#Drop irrelevant links from data:(e.g. newsletter,privacy-policy etc.)

links = pd.read_csv("bclinks.csv")

links = links.drop([0,len(links)-9,len(links)-8,len(links)-7,len(links)-6,len(links)-5,len(links)-4, len(links)-1,
                   len(links) - 3,len(links)-2],axis=0)

links.to_csv("bclinks.csv")

