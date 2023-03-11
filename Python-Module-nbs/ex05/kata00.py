kata = (19, 42, 21)

def display_numbers(numbers):
    num_str = ", ".join(str(num) for num in numbers)
    print(f"The {len(numbers)} numbers are: {num_str}")

if __name__ == "__main__":
    display_numbers(kata)
