import unittest
from datetime import datetime
from recipe import Recipe
from book import Book

class RecipeTest(unittest.TestCase):
    
    def test_recipe_attributes(self):
        r = Recipe("Pizza", 3, 45, ["dough", "cheese", "tomato sauce"], "A classic pizza recipe", "lunch")
        self.assertEqual(r.name, "Pizza")
        self.assertEqual(r.cooking_lvl, 3)
        self.assertEqual(r.cooking_time, 45)
        self.assertEqual(r.ingredients, ["dough", "cheese", "tomato sauce"])
        self.assertEqual(r.description, "A classic pizza recipe")
        self.assertEqual(r.recipe_type, "lunch")
        
    def test_recipe_type_error(self):
        with self.assertRaises(ValueError):
            r = Recipe("Burger", 2, 30, ["bread", "meat", "lettuce"], "A classic burger recipe", "dinner")
        
    def test_recipe_name_error(self):
        with self.assertRaises(TypeError):
            r = Recipe(123, 2, 30, ["bread", "meat", "lettuce"], "A classic burger recipe", "lunch")
        
    def test_recipe_cooking_lvl_error(self):
        with self.assertRaises(ValueError):
            r = Recipe("Burger", 6, 30, ["bread", "meat", "lettuce"], "A classic burger recipe", "lunch")
        
    def test_recipe_cooking_time_error(self):
        with self.assertRaises(ValueError):
            r = Recipe("Burger", 2, -10, ["bread", "meat", "lettuce"], "A classic burger recipe", "lunch")
        
    def test_recipe_ingredients_error(self):
        with self.assertRaises(TypeError):
            r = Recipe("Burger", 2, 30, "bread, meat, lettuce", "A classic burger recipe", "lunch")
        
    def test_recipe_description_error(self):
        with self.assertRaises(TypeError):
            r = Recipe("Burger", 2, 30, ["bread", "meat", "lettuce"], 123, "lunch")
        
        
class BookTest(unittest.TestCase):
    
    def setUp(self):
        self.book = Book("My Cookbook")
        
    def test_book_attributes(self):
        self.assertEqual(self.book.name, "My Cookbook")
        self.assertIsInstance(self.book.last_update, datetime)
        self.assertIsInstance(self.book.creation_date, datetime)
        self.assertEqual(len(self.book.recipes_list["starter"]), 0)
        self.assertEqual(len(self.book.recipes_list["lunch"]), 0)
        self.assertEqual(len(self.book.recipes_list["dessert"]), 0)
        
    def test_add_recipe(self):
        r = Recipe("Pizza", 3, 45, ["dough", "cheese", "tomato sauce"], "A classic pizza recipe", "lunch")
        self.book.add_recipe(r)
        self.assertEqual(len(self.book.recipes_list["lunch"]), 1)
        self.assertEqual(self.book.recipes_list["lunch"][0].name, "Pizza")
        self.assertIsInstance(self.book.last_update, datetime)
        
    def test_add_recipe_error(self):
        with self.assertRaises(AttributeError):
            self.book.add_recipe("not a recipe object")
        
    def test_get_recipe_by_name(self):
        r = Recipe("Pizza", 3, 45, ["dough", "cheese", "tomato sauce"], "A classic pizza recipe", "lunch")
        self.book.add_recipe(r)
        self.assertEqual(self.book.get_recipe_by_name("Pizza").name, "Pizza")
        self.assertIsNone(self.book.get_recipe_by_name("Pizzaaaa"))


if __name__ == '__main__':
    unittest.main()