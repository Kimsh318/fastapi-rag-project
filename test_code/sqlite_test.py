import sqlite3
import time
import random
import string
import os
import timeit
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import os


def time_stuff(some_function):
    def wrapper(*args, **kwargs):
        t0 = timeit.default_timer()
        value = some_function(*args, **kwargs)
        print(timeit.default_timer() - t0, 'seconds')
        return value
    return wrapper

def generate_values(count=100):
    end = int(time.time()) - int(time.time()) % 900
    symbol = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    ts = list(range(end - count * 900, end, 900))
    for i in range(count):
        yield (symbol, ts[i], random.random() * 1000, random.random() * 1000, random.random() * 1000, random.random() * 1000, random.random() * 1e9, random.random() * 1e5)

def generate_values_list(symbols=1000,count=100):
    values = []
    for _ in range(symbols):
        values.extend(generate_values(count))
    return values

@time_stuff
def sequential_read():
    """
    Read rows one after the other from a single thread
    100k records in the database, 1000 symbols, 100 rows
    First run
    0.25139795300037804 seconds
    Second run
    Third run
    """
    conn = sqlite3.connect(os.path.realpath('../files/ohlc.db'))
    try:
        with conn:
            conn.execute(create_statement)
            results = conn.execute(select_statement).fetchall()
            print(len(results))
    except sqlite3.OperationalError as e:
        print(e)