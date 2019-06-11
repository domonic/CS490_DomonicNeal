import requests
from bs4 import BeautifulSoup

'''Web scraping Write a simple program that parse a Wiki page mentioned and follow the instructions: https://en.wikipedia.org/wiki/Deep_learning
a. Print out the title of the page
b. Find all the links in the page (‘a’ tag)
c. Iterate over each tag(above) then return the link using attribute "href" using get'''

def get_webdata(url):
    web_page = requests.get(url)
    web_page_soup = BeautifulSoup(web_page.text, "html.parser")

    '''Output to screen the title of the page'''
    print("The title of this current web page is: ", web_page_soup.title.string)

    ''' Finds all links in the page (‘a’ tag) then iterates to then return the link using attribute "href" using get'''
    count = 0
    file = open("links.txt", "r+")

    for link in web_page_soup.find_all('a'):
        file.write(str(link.get('href')) + '\n')
        count += 1

    file.close()

    print("Links Found: ", count)
    print("All links successfully written to file...")

get_webdata("https://en.wikipedia.org/wiki/Deep_learning")








