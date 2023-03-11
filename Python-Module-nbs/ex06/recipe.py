cookbook = {
    'sandwich': {
        'ingredients': ['ham', 'bread', 'cheese', 'tomatoes'],
        'meal': 'lunch',
        'prep_time': 10
    },
    'cake': {
        'ingredients': ['flour', 'sugar', 'eggs'],
        'meal': 'dessert',
        'prep_time': 60
    },
    'salad': {
        'ingredients': ['avocado', 'arugula', 'tomatoes', 'spinach'],
        'meal': 'lunch',
        'prep_time': 15
    }
}

def print_recipe_names():
    print("Recipes in cookbook:")
    for recipe_name in cookbook:
        print(recipe_name)

def print_recipe(recipe_name):
    if recipe_name in cookbook:
        recipe = cookbook[recipe_name]
        print(f"Recipe for {recipe_name}:")
        print(f"Ingredients list: {recipe['ingredients']}")
        print(f"To be eaten for {recipe['meal']}.")
        print(f"Takes {recipe['prep_time']} minutes of cooking.")
    else:
        print(f"{recipe_name} recipe not found in cookbook.")

def add_recipe():
    recipe_name = input("Enter a name: ")
    ingredients = input("Enter ingredients, separated by commas: ").split(',')
    meal = input("Enter meal type: ")
    prep_time = int(input("Enter preparation time (in minutes): "))
    cookbook[recipe_name] = {
        'ingredients': ingredients,
        'meal': meal,
        'prep_time': prep_time
    }
    print(f"{recipe_name} recipe added to cookbook.")

def delete_recipe(recipe_name):
    if recipe_name in cookbook:
        del cookbook[recipe_name]
        print(f"{recipe_name} recipe deleted from cookbook.")
    else:
        print(f"{recipe_name} recipe not found in cookbook.")

def print_cookbook():
    print("Cookbook:")
    for recipe_name in cookbook:
        print_recipe(recipe_name)
        print()

def main():
    print("Welcome to the Python Cookbook!")
    while True:
        print("List of available options:")
        print("1: Add a recipe")
        print("2: Delete a recipe")
        print("3: Print a recipe")
        print("4: Print the cookbook")
        print("5: Quit\n")

        choice = input("Please select an option: ")
        if choice == '1':
            add_recipe()
        elif choice == '2':
            recipe_name = input("Please enter the name of the recipe to delete: ")
            delete_recipe(recipe_name)
        elif choice == '3':
            recipe_name = input("Please enter the name of the recipe to print: ")
            print_recipe(recipe_name)
        elif choice == '4':
            print_cookbook()
        elif choice == '5':
            print("Cookbook closed. Goodbye!")
            break
        else:
            print("Sorry, this option does not exist.")

if __name__ == '__main__':
    main()
