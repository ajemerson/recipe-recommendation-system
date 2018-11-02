import requests
import re
import pandas as pd
from bs4 import BeautifulSoup


def get_category_urls():
    """
    Finds the high-level categories of recipes; used for extracting URL patterns
    :return: dataframe with category names and associated urls. Columns will be ['type', url']
    """
    page = requests.get("https://www.allrecipes.com/recipes/")
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
        page = requests.get(row.url)
        content = page.content
        soup = BeautifulSoup(content, 'lxml')
        urls = soup.find('div', attrs={'class': 'grid slider'})
        for link in urls.find_all('a'):
            text = link.getText()
            text = text.replace('\n', '')
            sub_category_urls.loc[len(sub_category_urls)] = [row.type, text, link.get('href')]

        subtypes = subtypes.append(sub_category_urls)

    return subtypes


def example_recipe_scrape(type, subtype, url):
        """
        Sample code to produce a list of ingredients for a given recipe.
        The example recipe is 'Whole-wheat-pancakes-from-scratch' from the 'Pancakes' subcategory under the 'Breakfast and Brunch' category.
        :return: dataframe with columns ['type', 'subtype', 'recipe_title', 'ingredients']. The ingredient lists will need to be parsed.
        """
        final = pd.DataFrame(columns=['type', 'subtype', 'name', 'ingredients'])
        for i in range(2):
            example_sub_category = url + "?page=" + str(i + 1)
            print(example_sub_category)
            page = requests.get(example_sub_category)
            soup = BeautifulSoup(page.content, 'lxml')
            grid = soup.find('div', attrs={'class': 'fixed-grid'})
            articles = grid.find_all('article', attrs={'class': 'fixed-recipe-card'})
            j = 0
            for article in articles:
                ingredients = []
                example_recipe_url = article.find('a').get('href')
                recipe_title = article.find('span', attrs={'class':'fixed-recipe-card__title-link'}).getText()
                recipe_request = requests.get(example_recipe_url)
                recipe_soup = BeautifulSoup(recipe_request.content, 'lxml')
                ingredients_space = recipe_soup.find('div', attrs={'id': 'polaris-app'})
                ingredients_soup = ingredients_space.find_all('label', attrs={'ng-class': '{true: \'checkList__item\'}[true]'})

                for ingredient in range(len(ingredients_soup)):
                    ingredients.append(ingredients_soup[ingredient]['title'])

                j = j + 1
                final.loc[len(final)] = [type, subtype, recipe_title, ingredients]

        return final


if __name__ == "__main__":
    print("...Recipe Scraping begun...")

    categories = get_category_urls()
    sub_categories = get_subcategory_urls(categories)

    # Removing any instances where there is a faulty link/subtype
    sub_categories = sub_categories[sub_categories.subtype != '']
    sub_categories = sub_categories.reset_index()

    data = pd.DataFrame(columns=['type', 'subtype', 'name', 'ingredients'])
    for i in range(len(sub_categories)):
        temp = example_recipe_scrape(sub_categories.loc[i]['type'], sub_categories.loc[i]['subtype'], sub_categories.loc[i]['url'])
        data = data.append(temp)
    # print(data)

    data.to_csv('data.csv')
