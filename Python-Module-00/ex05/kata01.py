languages = {
    'Python': 'Guido van Rossum',
    'Ruby': 'Yukihiro Matsumoto',
    'PHP': 'Rasmus Lerdorf',
    'C++': 'Bjarne Stroustrup',
    }

s = ""
if languages:
    for key, value in languages.items():
        s += f"{key} was created by {value}\n"
else:
    s = "The dictionary is empty!"

print(s)