import pandas as pd
import os
import json
basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]

attendance_data = json.load(open(maindir+"\\Notebook_Scripts_Data\\data.json"))

attendance_data=json.dumps(attendance_data)
print(attendance_data)