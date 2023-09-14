import random


def guess():
    s = """This is an interactive guessing game!
You have to enter a number between 1 and 99 to find out the secret number.
Type 'exit' to end the game.
Good luck!"""
    print(s)
    n_of_attempts = 0
    random_num = random.randint(1, 100)
    while True:
        n_of_attempts += 1
        user_input = input("What's your guess between 1 and 99?")
        if user_input == "exit":
            print("Goodbye")
            break
        if user_input.isdigit() is False:
            print("That's not a valid input.")
            continue
        user_input = int(user_input)
        if user_input > random_num:
            print("Too high!")
        elif user_input < random_num:
            print("Too low!")
        elif user_input == 42 and n_of_attempts == 1:
            print("The answer to the ultimate question of life, the universe and everything is 42.\n"
                  "Congratulations! You got it on your first try!")
        else:
            print(f"Congratulations, you've got it!\n You won in {n_of_attempts} attempts!")
            break


if __name__ == '__main__':
    guess()