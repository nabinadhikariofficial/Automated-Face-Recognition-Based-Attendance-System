import json
from re import I

from matplotlib.font_manager import json_dump

data={
    "teacher":
    {
        "SS":{
            "name":"Siddhant Sharma",
            "subject":["Big Data"]
        },
        "PG":{
            "name":"Punam Gwachha",
            "subject":["Multimedia System"]
        },
        "AK":{
            "name":"Abhijit Karna",
            "subject":["Simulation and Modelling"]
        },
        "PS":{
            "name":"Prabesh Shrestha",
            "subject":["Internet and Intranet"]
        },
        "RP":{
            "name":"Rabindra Phonju",
            "subject":["Engineering Professional Practise"]
        },
        "SP":{
            "name":"Sajit Pyakurel",
            "subject":["Information System"]
        }
    },

    "student":{

    }
    ,


    "attendance":
    {
        "information system":[
            {
                "date": "",
                "student_username":[]
            },
        ],

        "big data":[
            {
                "date":"",
                "student_username":[]
            },
        ],
        "Simulation and modelling":[
            {
                "date":"" ,
                "student_username":[]
            },
        ],
        "engineering professional practise ":[
            {
                "date": "",
                "student_username":[]
            },
        ],

        "internet and intranet":[
            {
                "date":"" ,
                "student_username":[]
            },
        ],
        "multimedia system":[
            {
                "date":"" ,
                "student_username":[]
            }
        ]
        
    }
    

}






# with open('json_data.json') as json_file:
#     data = json.load(json_file)
#     # print(data)


# subject = {"Information System":{
#     "Present":0,
#     "total":0 
#     }
# }
# data["student"]["kce07bct001"]={"Information System":{
#     "Present":0,
#     "total":0
#     }
# }
# print(data["teacher"]["SS"]["name"])

i=1


while i < 49:
    if i <10:
        roll_no= "KCE074BCT00"+str(i)
    else:
        roll_no="KCE074BCT0"+str(i)
    
    data["student"][roll_no]={"Information System":{
        "Present":0,
        "total":0
        },
        "Multimedia System":{
        "Present":0,
        "total":0
        },
        "Engineering Professional Practise":{
        "Present":0,
        "total":0
        },
        "Big Data ":{
        "Present":0,
        "total":0
        },
        "Internet and Intranet":{
        "Present":0,
        "total":0
        },
        "Simulation and Modelling":{
        "Present":0,
        "total":0
        },
        
        

    }
    i=i+1

json_string =json.dumps(data)

with open('json_data.json', 'w') as outfile:
    outfile.write(json_string)


print(data)