import random

print("This is an interactive guessing game!")
print("You have to enter a number between 1 and 99 to find out the secret number.")
print("Type 'exit' to end the game.")
print("Good luck!")

secret_number = random.randint(1, 99)
num_guesses = 0

while True:
    guess = input("What's your guess between 1 and 99?\n")
    if guess == "exit":
        print("Goodbye!")
        break
    try:
        guess = int(guess)
    except ValueError:
        print("That's not a number.")
        continue

    num_guesses += 1

    if guess == secret_number:
        if secret_number == 42:
            print("The answer to the ultimate question of life, the universe and everything is 42.")
        print("Congratulations, you've got it!")
        if num_guesses == 1:
            print("You won in 1 attempt!")
        else:
            print(f"You won in {num_guesses} attempts!")
        break
    elif guess < secret_number:
        print("Too low!")
    else:
        print("Too high!")
