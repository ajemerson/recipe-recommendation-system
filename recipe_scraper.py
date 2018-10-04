import requests
import re
from bs4 import BeautifulSoup


def get_category_urls():
    page = requests.get("https://www.allrecipes.com/recipes/")
    content = page.content
    soup = BeautifulSoup(content)
    meal_types = soup.find('div', attrs = {'class': 'all-categories-col'})
    main_meals = meal_types.find('ul')
    categories = []
    for link in main_meals.find_all('a'):
        categories.append(link.get('href'))

    return categories


if __name__ == "__main__":
    print()

