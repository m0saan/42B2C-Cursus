import sys


def reverse(string: str):
    string = string.swapcase()
    return string[::-1]


if __name__ == '__main__':
    l = len(sys.argv[1:])
    for idx, s in enumerate(reversed(sys.argv[1:])):
        print(reverse(s), end='')
        print(" " if l != idx+1 else "", end='')
