t = (19, 42, 21)
if len(t) > 0:
    s = "The {} numbers are: {}, {}, {}".format(len(t), *t)
else:
    s = "The tuple is empty!"
print(s)