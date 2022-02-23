import pandas as pd
import os
import json
import time
from datetime import datetime
basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]
name="KCE074BCT001"
data= [{
        "date": "2022/02/21 12:39:29",
        "present_list": ["KCE074BCT023"],
        "absent_list": [
          "KCE074BCT001",
          "KCE074BCT002",
          "KCE074BCT003",
          "KCE074BCT004",
          "KCE074BCT005",
          "KCE074BCT006",
          "KCE074BCT007",
          "KCE074BCT008",
          "KCE074BCT009",
          "KCE074BCT010",
          "KCE074BCT011",
          "KCE074BCT012",
          "KCE074BCT013",
          "KCE074BCT014",
          "KCE074BCT015",
          "KCE074BCT016",
          "KCE074BCT017",
          "KCE074BCT018",
          "KCE074BCT019",
          "KCE074BCT020",
          "KCE074BCT021",
          "KCE074BCT022",
          "KCE074BCT024",
          "KCE074BCT025",
          "KCE074BCT026",
          "KCE074BCT027",
          "KCE074BCT028",
          "KCE074BCT029",
          "KCE074BCT030",
          "KCE074BCT031",
          "KCE074BCT032",
          "KCE074BCT033",
          "KCE074BCT034",
          "KCE074BCT035",
          "KCE074BCT036",
          "KCE074BCT037",
          "KCE074BCT038",
          "KCE074BCT039",
          "KCE074BCT040",
          "KCE074BCT041",
          "KCE074BCT042",
          "KCE074BCT043",
          "KCE074BCT044",
          "KCE074BCT045",
          "KCE074BCT046",
          "KCE074BCT047",
          "KCE074BCT048"
        ]
}
]

if (name in data[0]['absent_list']):
     print("true")
