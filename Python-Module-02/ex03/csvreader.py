class CsvReader:
    def __init__(self, filename=None, sep=',', header=False, skip_top=0, skip_bottom=0):
        self.filename = filename
        self.sep = sep
        self.header = header
        self.skip_top = skip_top
        self.skip_bottom = skip_bottom
        self.file = None
        
    def __enter__(self):
        try:
            self.file = open(self.filename, 'r')
            # he method returns self so that the class instance becomes the context manager. 
            # This means that when the with statement is executed and the __enter__ method is called, 
            # the class instance is returned, and any subsequent methods that are called on the instance within the with block can be executed.
            return self
        except FileNotFoundError:
            print("File not found.")
            return None
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
            
    def getdata(self):
        if self.file is None:
            return None
        
        data = []
        for i, line in enumerate(self.file):
            if i < self.skip_top:
                continue
            if i >= self.skip_top and i < self.get_total_lines() - self.skip_bottom:
                fields = line.strip().split(self.sep)
                if len(fields) != self.get_total_fields():
                    print(f"Line {i+1} is corrupted.")
                    return None
                data.append(fields)
        return data
    
    def getheader(self):
        if self.header and self.file is not None:
            header = self.file.readline().strip().split(self.sep)
            if len(header) != self.get_total_fields():
                print("Header is corrupted.")
                return None
            return header
        return None
    
    def get_total_lines(self):
        with open(self.filename, 'r') as f:
            return sum(1 for _ in f)
        
    def get_total_fields(self):
        with open(self.filename, 'r') as f:
            line = f.readline()
            return len(line.strip().split(self.sep))


if __name__ == "__main__":
    # with CsvReader('bad.csv', sep=';', header=True, skip_top=0, skip_bottom=0) as file:
    #     data = file.getdata()
    #     header = file.getheader()
        
    # if data is not None:
    #     # print(header)
    #     for row in data:
    #         print(row)
            
    # with CsvReader('bad.csv', sep=';', header=True, skip_top=1, skip_bottom=1) as file:
    #     if file is None:
    #         print("File is corrupted")
            
    with CsvReader('corrupted.csv', sep=';', header=True, skip_top=0, skip_bottom=0) as file:
        if file is None:
            print("File is corrupted")
        else:
            print('Not corrupted')
