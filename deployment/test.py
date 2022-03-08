from urllib.parse import uses_netloc
import pandas as pd
import os
import json
import time
from datetime import datetime
basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]

capture_bool = False

def g():
    global capture_bool
    capture_bool = True

print(capture_bool)
g()
print(capture_bool)
capture_bool = False
print(capture_bool)