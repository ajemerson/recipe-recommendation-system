import requests
import re
from bs4 import BeautifulSoup


def get_category_urls():
    """
    Finds the high-level categories of recipes; used for extracting URL patterns
    :return: high-level category names (e.g., 'breakfast')
    """
    page = requests.get("https://www.allrecipes.com/recipes/")
    content = page.content
    soup = BeautifulSoup(content)
    meal_types = soup.find('div', attrs={'class': 'all-categories-col'})
    main_meals = meal_types.find('ul')
    categories = []
    for link in main_meals.find_all('a'):
        categories.append(link.get('href'))

    return categories


# TODO
def scrape_categories(categories):
    """
    Extracts recipes for each of the given categories
    :param categories: high-level names
    :return: data structure of the overall recipes
    """
    for category in categories:
        sub_category_urls = []
        page = requests.get(category)
        content = page.content
        soup = BeautifulSoup(content)
        sub_categories = soup.find('div', attrs={'class': 'grid slider'})
        for link in sub_categories.find_all('a'):
            sub_category_urls.append(link.get('href'))


def example_recipe_scrape():
    """
    Sample code to produce a list of ingredients for a given recipe.
    The example recipe is 'Whole-wheat-pancakes-from-scratch' from the 'Pancakes' subcategory under the 'Breakfast and Brunch' category.
    :return: list of string ingredients. These need to be parsed.
    """
    example_sub_category = 'https://www.allrecipes.com/recipes/151/breakfast-and-brunch-pancakes/'
    page = requests.get(example_sub_category)
    soup = BeautifulSoup(page.content)
    grid = soup.find('div', attrs={'class': 'fixed-grid'})
    article = grid.find('article', attrs={'class': 'fixed-recipe-card'})
    example_recipe_url = article.find('a').get('href')
    recipe_request = requests.get(example_recipe_url)
    recipe_soup = BeautifulSoup(recipe_request.content)
    ingredients_space = recipe_soup.find('div', attrs={'id': 'polaris-app'})
    ingredients_soup = ingredients_space.find_all('label', attrs={'ng-class': '{true: \'checkList__item\'}[true]'})
    ingredients = []
    for ingredient in range(len(ingredients_soup)):
        ingredients.append(ingredients_soup[ingredient]['title'])

    return ingredients


if __name__ == "__main__":
    print("...Recipe Scraping begun...")
    # TODO: Automate the dataset creation in the main function.
