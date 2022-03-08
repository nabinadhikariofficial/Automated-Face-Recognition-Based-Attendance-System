from urllib.parse import uses_netloc
import pandas as pd
import os
import json
import time
from datetime import datetime
basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]
username="NABINa"
account = pd.read_csv(maindir+"\\Notebook_Scripts_Data\\accounts.csv", index_col=0).T
if username in account.columns:
    account=account[username].to_dict()
    print(account)
else:
    print(account)