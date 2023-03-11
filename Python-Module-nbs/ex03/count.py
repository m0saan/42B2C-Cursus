import string
import sys

def text_analyzer(text=None):
        
    if len(sys.argv) > 2:
        print(f'Error: more than one argument provided: {len(sys.argv)}')
        return
    elif len(sys.argv) == 2:
        text = sys.argv[1]
    
    if text is None:
        text = input('What is the text to analyze?\n')
    
    if not isinstance(text, str) or text.isdigit():
        raise AssertionError('Input text must be a string')
    
    upper_count = 0
    lower_count = 0
    punc_count = 0
    space_count = 0
    
    for char in text:
        if char.isupper():
            upper_count += 1
        elif char.islower():
            lower_count += 1
        elif char in string.punctuation:
            punc_count += 1
        elif char.isspace():
            space_count += 1
    
    print(f"- {upper_count} upper letter(s)")
    print(f"- {lower_count} lower letter(s)")
    print(f"- {punc_count} punctuation mark(s)")
    print(f"- {space_count} space(s)")

if __name__ == '__main__':
    text_analyzer()
