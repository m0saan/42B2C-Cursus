def what_are_the_vars(*args, **kwargs):
    obj = ObjectC()
    s = set()
    for i, arg in enumerate(args):
        var_name = f"var_{i}"
        if var_name in s:
            return None
        setattr(obj, var_name, arg)
        s.add(var_name)
        
    for k, v in kwargs.items():
        if k in s:
            return None
        setattr(obj, k, v)
    return obj

class ObjectC(object):
    def __init__(self):
        pass

def doom_printer(obj):
    if obj is None:
        print("ERROR")
        print("end")
        return
    for attr in dir(obj):
        if attr[0] != '_':
            value = getattr(obj, attr)
            print("{}: {}".format(attr, value))
    print("end")
    
if __name__ == "__main__":
    obj = what_are_the_vars(7)
    doom_printer(obj)

    obj = what_are_the_vars(None, [])
    doom_printer(obj)

    obj = what_are_the_vars("ft_lol", "Hi")
    doom_printer(obj)

    obj = what_are_the_vars()
    doom_printer(obj)

    obj = what_are_the_vars(12, "Yes", [0, 0, 0], a=10, hello="world")
    doom_printer(obj)

    obj = what_are_the_vars(42, a=10, var_0="world")
    doom_printer(obj)

    obj = what_are_the_vars(42, "Yes", a=10, var_2="world")
    doom_printer(obj)
