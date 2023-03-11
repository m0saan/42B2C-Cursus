import sys

def calculator():
    if len(sys.argv) == 1:
        print("""Usage: python operations.py <number1> <number2> \nExample:
        python operations.py 10 3""")
        return

    if len(sys.argv) != 3:
        print("AssertionError: too many arguments")
        return

    try:
        A = int(sys.argv[1])
        B = int(sys.argv[2])
    except ValueError:
        print("AssertionError: only integers")
        return
    
    is_div_by_zero = False
    # Perform operations

    sum = A + B
    diff = A - B
    prod = A * B
    try:
        quotient = A / B
        remainder = A % B
    except ZeroDivisionError: 
        is_div_by_zero = True

    # Print results
    print(f"Sum: {sum}")
    print(f"Difference: {diff}")
    print(f"Product: {prod}")
    if is_div_by_zero:
        print('ERROR (division by zero)')
        print('ERROR (modulo by zero)')
    else:
        print(f"Quotient: {quotient}")
        print(f"Remainder: {remainder}")

if __name__ == '__main__':
    calculator()
