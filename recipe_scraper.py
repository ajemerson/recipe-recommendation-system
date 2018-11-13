import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
import ast
import time


def get_category_urls():
    """
    Finds the high-level categories of recipes; used for extracting URL patterns
    :return: dataframe with category names and associated urls. Columns will be ['type', url']
    """
    page = ""
    while page == "":
        try:
            page = requests.get("https://www.allrecipes.com/recipes/")
            break
        except requests.exceptions.ConnectionError:
            print("Connection refused by server...")
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            continue
    content = page.content
    soup = BeautifulSoup(content, 'lxml')
    meal_types = soup.find('div', attrs={'class': 'all-categories-col'})
    main_meals = meal_types.find('ul')
    categories = pd.DataFrame(columns=['type', 'url'])
    for link in main_meals.find_all('a'):
        categories.loc[len(categories)] = [link.getText(), link.get('href')]

    return categories


def get_subcategory_urls(categories):
    """
    Extracts recipes for each of the given categories
    :param categories: dataframe of category names and urls. Columns should be ['type', 'url']
    :return: dataframe of type and subtype names and urls. Columns will be ['type', 'subtype', 'url']
    """
    subtypes = pd.DataFrame(columns=['type', 'subtype', 'url'])
    for index, row in categories.iterrows():
        sub_category_urls = pd.DataFrame(columns=['type', 'subtype', 'url'])
        page = ""
        while page == "":
            try:
                page = requests.get(row.url)
                break
            except requests.exceptions.ConnectionError:
                print("Connection refused by server...")
                print("Sleeping for 5 seconds...")
                time.sleep(5)
                continue
        content = page.content
        soup = BeautifulSoup(content, 'lxml')
        urls = soup.find('div', attrs={'class': 'grid slider'})
        for link in urls.find_all('a'):
            text = link.getText()
            text = text.replace('\n', '')
            sub_category_urls.loc[len(sub_category_urls)] = [row.type, text, link.get('href')]

        subtypes = subtypes.append(sub_category_urls)

    return subtypes


def generate_dataset(type, subtype, url):
        """
        Scrapes all recipes associated with each subcategory passed to the function.
        :param type: e.g., Breakfast and Brunch
        :param subtype: e.g., Pancakes
        :param url: subtype url - will loop through each page
        :return: dataframe with columns ['type', 'subtype', 'recipe_title', 'ingredients']. The ingredient lists will need to be parsed.
        """
        final = pd.DataFrame(columns=['type', 'subtype', 'name', 'ingredients'])
        page_count = get_last_page(url)
        for i in range(1, page_count):
            sub_category = url + "?page=" + str(i)
            print(sub_category)
            page = ""
            while page == "":
                try:
                    page = requests.get(sub_category)
                    break
                except requests.exceptions.ConnectionError:
                    print("Connection refused by server...")
                    print("Sleeping for 5 seconds...")
                    time.sleep(5)
                    continue
            soup = BeautifulSoup(page.content, 'lxml')
            grid = soup.find('div', attrs={'class': 'fixed-grid'})
            articles = grid.find_all('article', attrs={'class': 'fixed-recipe-card'})
            j = 0
            for article in articles:
                ingredients = []
                recipe_url = article.find('a').get('href')
                recipe_title = article.find('span', attrs={'class': 'fixed-recipe-card__title-link'}).getText()
                recipe_request = ""
                while recipe_request == "":
                    try:
                        recipe_request = requests.get(recipe_url)
                        break
                    except requests.exceptions.ConnectionError:
                        print("Connection refused by server...")
                        print("Sleeping for 5 seconds...")
                        time.sleep(5)
                        continue
                recipe_soup = BeautifulSoup(recipe_request.content, 'lxml')
                ingredients_space = recipe_soup.find('div', attrs={'id': 'polaris-app'})
                ingredients_soup = ingredients_space.find_all('label', attrs={'ng-class': '{true: \'checkList__item\'}[true]'})

                for ingredient in range(len(ingredients_soup)):
                    ingredients.append(ingredients_soup[ingredient]['title'])

                j = j + 1
                final.loc[len(final)] = [type, subtype, recipe_title, ingredients]

        return final


def get_last_page(url):
    """
    Finds the last page available for a given subtype of recipe
    :param url: subtype url. e.g., for Pancakes
    :return: integer (number) corresponding to the last page available
    """
    page_number = 1
    page_request = requests.get(url + "?page=" + str(page_number))
    while page_request.status_code == 200:
        page_number = page_number + 1
        page_request = ""
        while page_request == "":
            try:
                page_request = requests.get(url + "?page=" + str(page_number))
                break
            except requests.exceptions.ConnectionError:
                print("Connection refused by server...")
                print("Sleeping for 5 seconds...")
                time.sleep(5)
                continue

    return page_number


if __name__ == "__main__":
    print("...Recipe Scraping begun...")

    categories = get_category_urls()
    sub_categories = get_subcategory_urls(categories)

    # Removing any instances where there is a faulty link/subtype
    sub_categories = sub_categories[sub_categories.subtype != '']
    sub_categories = sub_categories.reset_index()

    data = pd.DataFrame(columns=['type', 'subtype', 'name', 'ingredients'])
    for i in range(16, len(sub_categories)):
        # Will save data each iteration and print out where the scraper left off in case of connection error.
        print(i)
        temp = generate_dataset(sub_categories.loc[i]['type'], sub_categories.loc[i]['subtype'], sub_categories.loc[i]['url'])
        data = data.append(temp)
        data.to_csv('data2.csv')

