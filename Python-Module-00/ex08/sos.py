import sys

morse_code = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    "0": "-----"
}


def string_to_morse_code(s: str) -> str:
    mc_str = ""
    for ch in s:
        if ch.isspace():
            mc_str += "/ "
            continue
        if not ch.isalpha() and not ch.isalnum():
            print("ERROR")
            exit(1)
        mc_str += morse_code[ch.upper()]
        mc_str += " "
    return mc_str


if __name__ == '__main__':
    morse_code_str = ""
    i = 0
    if len(sys.argv) >= 2:
        for arg in sys.argv[1:]:
            morse_code_str += string_to_morse_code(arg)
            i += 1
            if i < len(sys.argv)-1:
                morse_code_str += "/ "

    print(morse_code_str, end="")
