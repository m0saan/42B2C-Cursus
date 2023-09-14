import sys


def reverse(string: str):
    string = string.swapcase()
    print(string[::-1], end=" ")


if __name__ == '__main__':
    for s in reversed(sys.argv[1:]):
        reverse(s)
