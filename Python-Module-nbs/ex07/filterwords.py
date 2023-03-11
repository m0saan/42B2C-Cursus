import string
import sys

if len(sys.argv) != 3:
    print("ERROR")
else:
    try:
        n = int(sys.argv[2])
        s = sys.argv[1]
        words = s.split()
        filtered_words = [word.strip(string.punctuation) for word in words if len(word.strip(string.punctuation)) > n]
        print(filtered_words)
    except ValueError:
        print("ERROR")
