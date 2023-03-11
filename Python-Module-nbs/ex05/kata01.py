kata = {
'Python': 'Guido van Rossum',
'Ruby': 'Yukihiro Matsumoto',
'PHP': 'Rasmus Lerdorf',
}

def display_dict():
    for key,value in kata.items():
        print(f"{key} was was created by {value}")
        
        
        
if __name__ == '__main__':
    display_dict()
    