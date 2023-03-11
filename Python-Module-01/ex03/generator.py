import random

def my_shuffle(lst):
    n = len(lst)
    for i in range(n):
        j = random.randint(i, n-1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst


def generator(text, sep=" ", option=None):
    # Check if text is a string, return ERROR if not
    if not isinstance(text, str):
        yield "ERROR"
        return

    # Split the text according to sep
    words = text.split(sep)

    # Perform the requested option on the list of words
    if option == "shuffle":
        my_shuffle(words)
    elif option == "unique":
        words = list(set(words))
    elif option == "ordered":
        words = sorted(words)
    elif option is not None:
        yield "ERROR"
        return

    # Yield the resulting words one by one
    for word in words:
        yield word

if __name__ == '__main__':
    text = "Le Lorem Ipsum est simplement du faux texte."
    text = "Lorem Ipsum Lorem Ipsum"
    for word in generator(text, sep=" ", option="unique"):
        print(word)    
