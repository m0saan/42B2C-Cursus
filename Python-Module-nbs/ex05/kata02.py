kata = (2019, 9, 25, 3, 30)

month = f"{kata[1]:02d}"
day = f"{kata[2]:02d}"
year = kata[0]
hour = f"{kata[3]:02d}"
minute = f"{kata[4]:02d}"

print(f"{month}/{day}/{year} {hour}:{minute}")
