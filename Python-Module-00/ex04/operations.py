import sys


def operations(n1, n2) -> str:
    quotient = ""
    remainder = ""
    

    sum = n1 + n2
    diff = n1 - n2
    product = n1 * n2
    if n2 == 0:
        quotient = "ERROR (div by zero)"
        remainder = "ERROR (modulo by zero)"
    else:
        quotient = n1 / n2
        remainder = n1 % n2
    return """    Sum:         {}
    Difference:  {}
    Product:     {}
    Quotient:    {}
    Remainder:   {}""".format(sum, diff, product, quotient, remainder)


if __name__ == '__main__':
    strHelp = """Example:
    python operations.py 10 3"""
    av = sys.argv
    if len(av) != 3:
        print(strHelp)
    elif len(av) == 3 and not av[1].lstrip('-+').replace('.', '', 1).isdigit() or \
        not av[2].replace('.', '', 1).lstrip('-+').isdigit():
        print("AssertionError: only integers")
    else:
        try:
            print(operations(int(av[1]), int(av[2])))
        except:
            print("AssertionError: no floats")
