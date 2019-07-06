########################################################################################
########################################################################################
###############    Extracting text/date from Coindesk links   ##########################
###############             and cleaning the data             ##########################
###############                                               ##########################
########################################################################################
########################################################################################


import re
from bs4 import BeautifulSoup as soup
import requests
import datetime
import pandas as pd
import csv
import pickle
import nltk
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt


link_list = []
content_list = []


#Convert bclinks.csv to list of links
with open("bclinks.csv", "rt") as f:   #,encoding= "utf-8"
    reader = csv.reader(f)
    link_list = list(reader)



#Get the content and timestamp of each article(by link_list)
news_text = []
time_list = []

for i in range(0,len(link_list)): #for loop from 0 to len(link_list) as links have been manually adjusted
    print("This is Article: ", i)
    btc_content_url = link_list[i][0]
    url_request = requests.get(btc_content_url)
    url_content = url_request.content

    parsed_content = soup(url_content)

    containers_a = parsed_content.find_all('p')
    news_text = "\n"
    for container in containers_a:
        news_text += container.text.strip()
    content_list.append(news_text)
    containers_b = parsed_content.find_all('div', attrs={'class':'timestamp'})[0]
    for container in containers_b.find_all('span'):
        news_time = datetime.datetime.strptime(container.text.strip(),'%b %d, %Y at %H:%M %Z') 
        time_list.append(news_time)



#Pickle the data for faster loading times:

#with open('content_list', 'wb') as f:
 #      pickle.dump(content_list, f)
#with open("time_list","wb") as g:
 #       pickle.dump(time_list, g)

# Open the data with pickle file
with open('content_list', 'rb') as f:
        content_list = pickle.load(f)
with open("time_list","rb") as g:
        time_list = pickle.load(g)

#content_list = pd.DataFrame(content_list)

#article_info = pd.DataFrame(content_list,time_list)

#article_info.to_csv("Coindesk_articles.csv")

#Plot the Data:
plt.style.use('seaborn') #Different styles possible like "ggplot" etc.
fig = plt.figure()
plt.plot(df_count)
fig.suptitle("Article frequency on Coindesk.com")
plt.xlabel('year', fontsize=10)
plt.ylabel('# articles', fontsize=8)
plt.savefig('Data_Coindesk.png')


#Converts every latter to lower except the ones at the beginning of a sentence
article_list = []
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
for article in content_list:

    article = re.sub('[^A-Za-z.!? ]+', '', article) #drops every non alphabetic/numeric or !?., character [0-9]
    article = article.lower() #Converts everything in news_text to lowercase
    article = re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), article)#Adds capital letter after a dot
    #news_text = re.sub(r'([A-Z])',r" \1",news_text,re.MULTILINE)  #Adds space before every capital letter
    article = re.sub(r'(?<=[.])(?=[^\s])', r' ', article)
    #Space not added in front of every capital letter!

    #Use a pretrained model to tokenize into sentences:
    article_list.append(tokenizer.tokenize(article))

#Save the content and timestamp of each article:
article_info = pd.DataFrame(article_list,time_list)



#with open('article_info', 'wb') as g:
#       pickle.dump(article_info, g)

with open('article_info', 'rb') as g:
    article_info = pickle.load(g)




