import nltk
import pandas as pd
import unit_conversion as uc
import numpy as np

delinquents = [8726, 8727, 13586, 13587, 22594, 22595, 26395, 26396, 19016, 19017, 23876, 23877, 32884, 32885, 36685,
               36686, 57900, 57944, 57945, 57946]

def temp():
    lemma = nltk.wordnet.WordNetLemmatizer()

    unit_corpus = list(set(['oz', 'ounces', 'ounce', 'gram', 'grams', 'ml', 'l', 'pound', 'lb',
            'ozs', 'stone', 'st', 's.t.', 'milliliters', 'ton', 't', 'micrograms',
            'microgram', 'kilograms', 'kg', 'kilogram', 'metric ton', 'mt', 'm.t',
            'metric tonne','liter', 'tsp', 'teaspoons', 'teaspoon', 'tbsp',
            'tablespoons', 'cup', 'cups', 'c', 'floz', 'fluid oz', 'fluid ounces',
            'quart', 'qu', 'qt', 'pint', 'pt', 'gallons', 'gal', 'tablespoon', 'tablespoons',
                           'pinch', 'as needed', 'tiny pinch', 'drops', 'gallon', 'drop', 'dash', 'dashes']))

    def remove_parens(string):
        while True:
            if '(' in string:
                left = string.index('(')
                right = string.index(')')
                if left > 0:
                    if right == len(string) - 1:
                        string = string[:left] # there is nothing to the right of the right parenthesis
                    elif right < len(string) - 2:
                        string = string[:left] + string[right+2:] # get rid of underscore after right parenthesis
                    elif right < len(string) - 1:
                        string = string[:left] + string[right+1:]
                elif left == 0:
                    string = string[right+1:] # there is nothing to the left of the left parenthesis
            elif ')' in string: # only needed when there is a set of parentheses within another
                ind = string.index(')')
                if ind == len(string) - 1:
                    string = string[:ind]
                else:
                    string = string[:ind] + string[ind+1:]
            else:
                break
            string = string.strip()
        return string

    def preprocess_ingredient(string):
        string = string.lower()
        if ',' in string:
            comma = string.index(',')
            # truncate column name to before the comma (assuming directions to the right)
            string = string[0:comma]
        return string


    def determine_unit(recipe):
        # Assume that the parameter, recipe, is a string of the entire recipe but has been fully preprocessed
        # for parentheses and commas

        # Loop the elements of the unit corpus through the recipe string
        # add spces to insure that the search does not simply find the instance of a character in the recipe
        size = len(unit_corpus)
        unit = ''
        for i in range(size):
            un = ' ' + unit_corpus[i] + ' '
            if un in recipe:
                unit = un.strip()
                break
        return (unit)


    def find_frac(cur):
        # Assume that the parameter, recipe, is a string of the entire recipe but has been fully preprocessed
        # for parentheses and commas

        try:
            first = cur[0]
            second = cur[1]
        except:
            first = cur[0]
            second = ''
            # print('Look at this:', first, cur)
        if '/' in first:
            # Then there is a fractional measurement in the first "word" of the recipe
            frac = first.split('/')  # split numerator from denominator
            num, denom = (frac[0], frac[1])
            # make sure / isn't just the 'or' symbol
            try:
                num = int(num)
                denom = int(denom)
                measurement = num / denom
                quantity = False
            # If it is, default the measurement to 1 'quantity'
            except:
                measurement = 1
                quantity = True
            cur = cur[1:]
            ingred = '_'.join(cur)
        elif '/' in second:  # the second word of the recipe is a fraction (e.g. 1 3/4 cups)
            frac = second.split('/')
            try:
                num, denom = (frac[0], frac[1])
                num = int(num)
                denom = int(denom)
                measurement = int(first) + num / denom
                quantity = False
            except:
                measurement = 1
                quantity = True
            cur = cur[2:]
            ingred = '_'.join(cur)
        else:
            # Check if the first word of the ingredient is even a number at all
            try:
                measurement = int(first)
                ingred = '_'.join(cur[1:])
                quantity = False
            except:
                # In this case, there is not a measurement for this ingredient
                # It is itself a unit, e.g. cooking spray
                measurement = 1
                quantity = True
                ingred = '_'.join(cur)
        return (ingred, measurement, quantity)


    def parse_recipe(recipe, dictionary={}, col_list=[], measurement_list=[], iteration=0):
        """
        Extract the measurement and unit of the ingredient as well as whether or not it should be expressed as a quantity
        An ingredient, for our purposes, is expressed as EITHER a measurement (e.g. 3 referring to cups) OR a quantity
        (e.g., 3 referring to eggs)
        Quantity will be set to True or False based on these observations
        :param recipe: an array where each entry is as follows: (measurement, unit, ingredient)
        """
        ings = len(recipe)
        for i in range(ings):
            # remove parentheses from ingredients
            recipe[i] = remove_parens(recipe[i])
            recipe[i] = preprocess_ingredient(recipe[i])
            cur = recipe[i].split()  # make the ingredient into an array itself!
            # lemmatize the ingredient specification
            for j in range(len(cur)):
                cur[j] = lemma.lemmatize(cur[j])
            #             if ',' in cur[j]:
            #                 comma = cur[j].index(',')
            #                 word = cur[j][:comma]
            recipe[i] = ' '.join(cur)
            ind = 0
            # If the first word is 'about', check the next word for the measurement
            if cur[ind].lower().strip() == 'about':
                ind += 1

            unit = determine_unit(recipe[i])
            if unit != '':
                recipe[i] = recipe[i].replace(unit + ' ', '')  # remove unit from recipe string
            cur = recipe[i].split()

            ingred, measurement, quantity = find_frac(cur)

            # If quantity is False, we do a unit conversion of the found measurement. However, if the found "unit"
            # (second or third word) does not match a unit in our dictionary, we express the ingredient as a quantty
            # This is checked in the convert function itself
            # We won't worry about making a dataframe just yet. We will first list all the columns and the corresponding values
            if quantity or unit == '':
                # print('quantity')
                # print('No conversion necessary')
                col_list.append(ingred)
                measurement_list.append(measurement)
            elif not quantity and unit != '':
                # print(unit)
                mes, cols, quantity = uc.convert(measurement, unit, ingred)
                if quantity:
                    col_list.append(ingred)
                    measurement_list.append(measurement)
                    # If we know for sure the unit we found is in our dictionary, we have a valid measurement
                elif not quantity:
                    col_list.extend(cols)
                    measurement_list.extend(mes)
        # lemmatize
        for i in range(len(col_list)):
            temp = col_list[i].split('_')
            for j in range(len(temp)):
                temp[j] = lemma.lemmatize(temp[j])
            col_list[i] = '_'.join(temp)

        # Now populate the dictionary
        # Either add a value to the list of values corresponding to a key
        # Or create a new key value pair
        # init key value pairs as col_list and empty lists per column
        if len(dictionary) == 0:
            dictionary = dict(zip(col_list, [[] for i in range(len(col_list))]))

        for i in set(col_list + list(dictionary.keys())):
            if i in col_list:
                if i in dictionary:
                    dictionary[i] += [measurement_list[col_list.index(i)]]
                else:
                    dictionary[i] = [0] * iteration + [measurement_list[col_list.index(i)]]
            else:
                if i in dictionary:
                    dictionary[i] += [0]
                else:
                    dictionary[i] = [0]
        # Will be used to construct dataframe
        return dictionary

    data = pd.read_csv('Data/preparsed_data.csv')
    import ast
    ingz = []
    for i in range(len(data)):
        ingz.append(ast.literal_eval(data['ingredients'][i]))
    size = len(ingz)
    print('Number of recipes:', size)

    d = {}
    c_list = []
    for i in range(0, size):  # read one row from csv file at a time
        if i not in delinquents:
            rec = ingz[i]
            # print('Recipe:', rec)
            # print(ing)
            # for j in range(size):
            # rec = ingz[i]
            # print(rec)
            d = parse_recipe(recipe=rec, dictionary=d, col_list=[], measurement_list=[], iteration=i)
            # c_list.extend(c)
            # c_list = list(set(c_list))
            # print('Size of column list:', len(c_list))
            # print('\n')
            if i < max(delinquents):  # debugging
                if i % 100 == 0:
                    print(i)
            else:
                print(i)
        else:
            cols = list(d.keys())
            for j in cols:
                d[j] += [0]
    print(len(d))
    print(len(d.values()))
    print(len(d.keys()))
    print('Saving dataframe.......................')
    # print(len(c_list))

    # c_list.insert(0, 'subtype')
    # c_list.insert(0, 'type')
    # c_list.insert(0, 'name')

    # with open(path + 'parsed_data.csv', 'a') as colfile:
    #     wr = csv.writer(colfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(c_list)
    #
    # print('Column list has been saved to CSV :)')

    df = pd.DataFrame(data=np.array(list(d.values())).T, columns=np.array(list(d.keys())))

    print("Appending additional columns.....")
    additional_columns(data, df)

    # Split the data frame so that we guarantee enough space
    # df1, df2, df3, df4, df5 = np.array_split(df, 5)
    #
    # df1.to_csv('/Users/nathanpool/Desktop/parsed_data1.csv')
    # df2.to_csv('/Users/nathanpool/Desktop/parsed_data2.csv')
    # df3.to_csv('/Users/nathanpool/Desktop/parsed_data3.csv')
    # df4.to_csv('/Users/nathanpool/Desktop/parsed_data4.csv')
    # df5.to_csv('/Users/nathanpool/Desktop/parsed_data5.csv')

    print('YEET. YOU DONE, DOG.')


def additional_columns(data, df):
    path = '/Users/nathanpool/Desktop/'
    # print('Reading csv...', path, 'preparsed_data', str(j), '.csv')
    # data = pd.read_csv(path + 'preparsed_data' + str(j) + '.csv')
    # print('csv was saved to dataframe.')
    length = len(data)
    subdf = pd.DataFrame(columns=["name", "type", "subtype"])
    print('Gathering names, types, etc. into a subdataframe...')
    for i in range(length):
        # Ignore the delinquent columns
        if i not in delinquents:
            name = data.iloc[i, 3]
            type = data.iloc[i, 1]
            subtype = data.iloc[i, 2]
            subdf.loc[i] = [name, type, subtype]
        else:
            name = "Delete"
            type = "Delete"
            subtype = "Delete"
            subdf.loc[i] = [name, type, subtype]
        if i < max(delinquents):  # debugging
            if i % 100 == 0:
                print(i)
        else:
            print(i)

    print('Names gathered! ')

    print('Appending...')

    append_dataframes(path, subdf, df)


def append_dataframes(path, subdf, og):
    add_cols = subdf  # dataframe with name, type, and subtype
    # og_frame = pd.read_csv(path + 'parsed_data' + str(j) + '.csv')
    og_frame = og  # the parsed dataframe

    print('Concatenating...')
    result = pd.concat([add_cols, og_frame], axis=1, sort=False)

    # print('Removing delinquent rows...')
    # result.drop(result.index[[8726, 13586, 22594, 26395, 26396]], inplace=True)
    # result.reset_index()

    print("Preview of the data...")
    print(result[['name', 'type', 'subtype']].head(10))

    print('Saving concatenated dataframe.....')

    result.to_csv(path + 'final_data.csv')

    print('YEET')


if __name__ == '__main__':
    temp()
    # additional_columns()  --- this is already called from temp()