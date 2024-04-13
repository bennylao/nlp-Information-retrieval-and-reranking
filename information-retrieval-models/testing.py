import os
from timeit import default_timer as timer


start = timer()

os.system("python task1.py")
os.system("python task2.py")
os.system("python task3.py")
os.system("python task4.py")

end = timer()
time_taken = end - start

print(f"Total time: {time_taken} seconds")
