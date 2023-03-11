class Recipe:
    def __init__(self, name, cooking_lvl, cooking_time, ingredients, description, recipe_type):
        self.name = name
        self.cooking_lvl = cooking_lvl
        self.cooking_time = cooking_time
        self.ingredients = ingredients
        self.description = description
        self.recipe_type = recipe_type
        
        if not isinstance(self.name, str):
            raise TypeError("The name of the recipe must be a string.")
        if not isinstance(self.cooking_lvl, int) or not 1 <= self.cooking_lvl <= 5:
            raise ValueError("The cooking level must be an integer between 1 and 5.")
        if not isinstance(self.cooking_time, int) or self.cooking_time < 0:
            raise ValueError("The cooking time must be a positive integer.")
        if not isinstance(self.ingredients, list) or not all(isinstance(i, str) for i in self.ingredients):
            raise TypeError("The ingredients must be a list of strings.")
        if not isinstance(self.description, str):
            raise TypeError("The description must be a string.")
        if self.recipe_type not in ["starter", "lunch", "dessert"]:
            raise ValueError("The recipe type must be 'starter', 'lunch' or 'dessert'.")
        
    def __str__(self):
        """Return the string to print with the recipe info"""
        txt = f"{self.name}, Cooking level: {self.cooking_lvl}, Cooking time: {self.cooking_time} minutes,\n"
        txt += "Ingredients: " + ", ".join(self.ingredients) + "\n"
        if self.description:
            txt += f"Description: {self.description}\n"
        txt += f"Type: {self.recipe_type}"
        return txt
