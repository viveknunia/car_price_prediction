from selenium import webdriver
from selenium.common import exceptions 
import csv
import sys
import time
import io
def scrape(url,filename):
    """
    Extracts the comments from the Youtube video given by the URL.
    Args:
        url (str): The URL to the Youtube video
    Raises:
        selenium.common.exceptions.NoSuchElementException:
        When certain elements to look for cannot be found
    """

    # Note: Download and replace argument with path to the driver executable.
    # Simply download the executable and move it into the webdrivers folder.
    driver = webdriver.Chrome('webdrivers/chromedriver')

    # Navigates to the URL, maximizes the current window, and
    # then suspends execution for (at least) 5 seconds (this
    # gives time for the page to load).
    driver.get(url)
    driver.maximize_window()
    time.sleep(5)

    try:
        # Extract the elements storing the video title and
        # comment section.
        title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
        comment_section = driver.find_element_by_xpath('//*[@id="comments"]')
    except exceptions.NoSuchElementException:
        # Note: Youtube may have changed their HTML layouts for
        # videos, so raise an error for sanity sake in case the
        # elements provided cannot be found anymore.
        error = "Error: Double check selector OR "
        error += "element may not yet be on the screen at the time of the find operation"
        print(error)

    # Scroll into view the comment section, then allow some time
    # for everything to be loaded as necessary.
    driver.execute_script("arguments[0].scrollIntoView();", comment_section)
    time.sleep(7)

    # Scroll all the way down to the bottom in order to get all the
    # elements loaded (since Youtube dynamically loads them).
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        # Scroll down 'til "next load".
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

        # Wait to load everything thus far.
        time.sleep(2)

        # Calculate new scroll height and compare with last scroll height.
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # One last scroll just in case.
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

    try:
        # Extract the elements storing the usernames and comments.
        username_elems = driver.find_elements_by_xpath('//*[@id="author-text"]')
        comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
    except exceptions.NoSuchElementException:
        error = "Error: Double check selector OR "
        error += "element may not yet be on the screen at the time of the find operation"
        print(error)

    print("> VIDEO TITLE: " + title + "\n")

    with io.open('vid_comments_'+str(filename)+'.csv', 'w', newline='', encoding="utf-16") as file:
        writer = csv.writer(file, delimiter =",", quoting=csv.QUOTE_ALL)
        writer.writerow(["Username", "Comment"])
        for username, comment in zip(username_elems, comment_elems):
            writer.writerow([username.text, comment.text])

    driver.close()


l = []

while(True):
    temp = input('Enter URL or type break to end ')
    if temp == 'break':
        break
    else:
        l.append(temp)

for i in range(len(l)):
    scrape(l[i],i)