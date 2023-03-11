from datetime import datetime
from recipe import Recipe

class Book:
    def __init__(self, name):
        self.name = name
        self.last_update = datetime.now()
        self.creation_date = datetime.now()
        self.recipes_list = {"starter": [], "lunch": [], "dessert": []}

    def add_recipe(self, recipe):
        if not isinstance(recipe, Recipe):
            raise AttributeError("Error: Can only add Recipe objects to Book")

        self.recipes_list[recipe.recipe_type].append(recipe)
        self.last_update = datetime.now()

    def get_recipe_by_name(self, name):
        for recipe_type in self.recipes_list:
            for recipe in self.recipes_list[recipe_type]:
                if recipe.name == name:
                    return recipe
        print(f"No recipe found with name '{name}'")
        return None

    def get_recipes_by_types(self, recipe_type):
        recipe_names = []
        for recipe in self.recipes_list[recipe_type]:
            recipe_names.append(recipe.name)
        return recipe_names

    def __str__(self):
        txt = f"Book: {self.name}\n"
        for recipe_type in self.recipes_list:
            txt += f"\n{recipe_type.title()} recipes:\n"
            for recipe in self.recipes_list[recipe_type]:
                txt += f" - {recipe.name}\n"
        return txt
