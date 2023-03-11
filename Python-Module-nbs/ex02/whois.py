import sys


import sys

def whois():
    if len(sys.argv) < 2:
        print("Usage: python <filename.py> <integer>")
        return
        
    if len(sys.argv) > 2:
        print(f'AssertionError: more than one argument provided: {len(sys.argv)}')
        return
        
    if not sys.argv[1].isdigit():
        print(f'AssertionError: argument is not an integer')
        return
        
    number = int(sys.argv[1])
    
    if number == 0:
        print("I'm Zero")
    elif number % 2 == 0:
        print("I'm Even")
    else:
        print("I'm Odd")

        


if __name__ == '__main__':
    whois()
 