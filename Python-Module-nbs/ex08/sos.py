import string

MORSE_TABLE = {
    'A': '.-',     'B': '-...',   'C': '-.-.', 
    'D': '-..',    'E': '.',      'F': '..-.',
    'G': '--.',    'H': '....',   'I': '..',
    'J': '.---',   'K': '-.-',    'L': '.-..',
    'M': '--',     'N': '-.',     'O': '---',
    'P': '.--.',   'Q': '--.-',   'R': '.-.',
    'S': '...',    'T': '-',      'U': '..-',
    'V': '...-',   'W': '.--',    'X': '-..-',
    'Y': '-.--',   'Z': '--..',
    '0': '-----',  '1': '.----',  '2': '..---',
    '3': '...--',  '4': '....-',  '5': '.....',
    '6': '-....',  '7': '--...',  '8': '---..',
    '9': '----.'
}

def encode_morse(string):
    encoded = []
    for char in string.upper():
        if char == ' ':
            encoded.append('/')
        elif char in MORSE_TABLE:
            encoded.append(MORSE_TABLE[char])
        else:
            print('ERROR'); exit()
    return ' '.join(encoded)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 sos.py 'string to encode'")
    else:
        input_str = ' '.join(sys.argv[1:])
        print(encode_morse(input_str))
