import time
from random import randint
import os

#... your definition of log decorator...
def log(func: callable):
    def wrapper(*args, **kwargs):
        with open('machone.log', 'a') as f:
            username = os.environ.get('USER')
            start = time.time()
            func_name  = [word.capitalize() for word in func.__name__.split('_')]
            func_name = ' '.join(func_name)
            
            func(*args, **kwargs)
            
            end = time.time()
            exec_time_ms = (end - start) * 1000
            
            f.write(f'({username}) Running: {func_name :<–20} [ exec-time = {exec_time_ms:.2f} ms ]\n')
    return wrapper


class CoffeeMachine():
    
    water_level = 100

    @log
    def start_machine(self):
        if self.water_level > 20:
            return True
        else:
            print("Please add water!")
            return False

    @log
    def boil_water(self):
        return "boiling..."

    @log
    def make_coffee(self):
        if self.start_machine():
            for _ in range(20): time.sleep(0.1)
            self.water_level -= 1
            print(self.boil_water())
            print("Coffee is ready!")
            
    @log
    def add_water(self, water_level):
        time.sleep(randint(1, 5))
        self.water_level += water_level
        print("Blub blub blub...")
        

if __name__ == "__main__":
    machine = CoffeeMachine()
    for i in range(0, 5):
        machine.make_coffee()
        machine.make_coffee()
        machine.add_water(70)