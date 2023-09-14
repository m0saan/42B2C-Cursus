import sys


def filter_words(s: str, n: int) -> list:
    punctuation_marks = "!#$%&'()*+, -./:;<=>?@[\]^_`{|}~"
    for ch in s:
        if ch in punctuation_marks:
            s = s.replace(",", "")
    words = s.split(" ")
    ls = [word for word in words if len(word) > n and word not in punctuation_marks]
    return ls


if __name__ == '__main__':
    av = sys.argv
    if (len(av) != 3 or av[1].isdigit() or not av[2].isdigit()):
        print("ERROR")
        exit()
    else:
        print(filter_words(av[1], int(av[2])))
