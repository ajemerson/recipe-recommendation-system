import requests
import re
import pandas as pd
from bs4 import BeautifulSoup


def get_category_urls():
    """
    Finds the high-level categories of recipes; used for extracting URL patterns
    :return: high-level category names (e.g., 'breakfast')
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


# TODO
'''def scrape_categories(categories):
    """
    Extracts recipes for each of the given categories
    :param categories: high-level names
    :return: data structure of the overall recipes
    """
    for category in categories:
        sub_category_urls = []
        page = requests.get(category)
        content = page.content
        soup = BeautifulSoup(content, 'lxml')
        sub_categories = soup.find('div', attrs={'class': 'grid slider'})
        for link in sub_categories.find_all('a'):
            sub_category_urls.append(link.get('href'))

        return sub_category_urls'''


def all_recipe_scrape(type, url):
        """
        Sample code to produce a list of ingredients for a given recipe.
        The example recipe is 'Whole-wheat-pancakes-from-scratch' from the 'Pancakes' subcategory under the 'Breakfast and Brunch' category.
        :return: list of string ingredients. These need to be parsed.
        """
        all_final_data = pd.DataFrame(columns=['type', 'name', 'ingredients'])
        for i in range(2):
            example_sub_category = url + "?page=" + str(i)
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
                all_final_data.loc[len(all_final_data)] = [type, recipe_title, ingredients]

        return all_final_data


if __name__ == "__main__":
    print("...Recipe Scraping begun...")
    # TODO: Automate the dataset creation in the main function.

    categories = get_category_urls()
    data_all = pd.DataFrame(columns=['type', 'name', 'ingredients'])
    for i in range(len(categories)):
        temp = all_recipe_scrape(categories.loc[i]['type'], categories.loc[i]['url'])
        data_all = data_all.append(temp)
    print(data_all)
    data_all.to_csv('data.csv')
