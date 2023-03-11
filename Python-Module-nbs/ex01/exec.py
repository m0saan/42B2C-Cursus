import sys

def rev_alpha(str_input=None):
    
    """
    Reverses the order of the words in a sentence, and swaps the case of each character.
    
    This function reads the command-line arguments passed to the script (excluding the script name itself),
    merges them into a single string separated by spaces, and then reverses the order of the words in the string.
    Finally, the function swaps the case of each character in the resulting string and prints it to the console.
    
    Args:
        None
        
    Returns:
        None
    
    Example:
        If the script is invoked with the command "python rev_alpha.py Hello, world!",
        the output will be "!DLROW ,OLLEh"
    """
    
    merged_args = ' '.join(str_input if str_input is not None else sys.argv[1:])
    reversed_str = merged_args[::-1].swapcase()
    print(reversed_str)



if __name__ == '__main__':
    rev_alpha()
