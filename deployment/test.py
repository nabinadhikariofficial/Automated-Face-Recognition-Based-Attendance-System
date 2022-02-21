import pandas as pd
import os
import json
import time
from datetime import datetime
basedir = os.path.abspath(os.path.dirname(__file__))
maindir = basedir[:-11]
data=[['KCE074BCT001', 'AAKASH RAJ DHAKAL', 'Absent'], ['KCE074BCT002', 'AAKASH SHRESTHA', 'Absent'], ['KCE074BCT003', 'AAKRITI AGANJA', 'Absent'], ['KCE074BCT004', 'AAYUSH MUSYAJU', 'Absent'], ['KCE074BCT005', 'ABHINAV ARYAL', 'Absent'], ['KCE074BCT006', 'ABHISHEK NEUPANE', 'Absent'], ['KCE074BCT007', 'AJAYA THAKUR', 'Absent'], ['KCE074BCT008', 'AMAR SUNAR', 'Absent'], ['KCE074BCT009', 'ANISH NEUPANE', 'Absent'], ['KCE074BCT010', 'ANKIT PRADHAN', 'Absent'], ['KCE074BCT011', 'ANUSHA BAJRACHARYA', 'Absent'], ['KCE074BCT012', 'ASMIN KARKI', 'Absent'], ['KCE074BCT013', 'BASANT BABU BHANDARI', 'Absent'], ['KCE074BCT014', 'BIBASH RAJTHALA', 'Absent'], ['KCE074BCT015', 'BIBEK SHYAMA', 'Absent'], ['KCE074BCT016', 'BIKESH SITIKHU', 'Absent'], ['KCE074BCT017', 'CHIRANJEVI UPADHYAYA', 'Absent'], ['KCE074BCT018', 'KIRAN NEUPANE', 'Absent'], ['KCE074BCT019', 'KUSHAL SUWAL', 'Absent'], ['KCE074BCT020', 'LAXMAN MAHARJAN', 'Absent'], ['KCE074BCT021', 'LUCKY SINGTON SHRESTHA', 'Absent'], ['KCE074BCT022', 'LUJA SHAKYA', 'Absent'], ['KCE074BCT023', 'NABIN ADHIKARI', 'Absent'], ['KCE074BCT024', 'NIRAJAN PRAJAPATI', 'Absent'], ['KCE074BCT025', 'NIRANJAN BEKOJU', 'Absent'], ['KCE074BCT026', 'NIRJAL PRAJAPATI', 'Absent'], ['KCE074BCT027', 'OM KRISHNA RAUT', 'Present'], ['KCE074BCT028', 'OSHIN GANSI', 'Absent'], ['KCE074BCT029', 'PRASANNA ADHIKARI', 'Absent'], ['KCE074BCT030', 'PRASHUN CHITRAKAR', 'Absent'], ['KCE074BCT031', 'RAM KATWAL', 'Absent'], ['KCE074BCT032', 'RATISH NYAICHYAI', 'Absent'], ['KCE074BCT033', 'ROHIT PRAJAPATI', 'Absent'], ['KCE074BCT034', 'ROJASH SHAHI', 'Absent'], ['KCE074BCT035', 'SABIN SUWAL', 'Absent'], ['KCE074BCT036', 'SACHIT KUMAR SHRESTHA', 'Absent'], ['KCE074BCT037', 'SAHAS PRAJAPATI', 'Absent'], ['KCE074BCT038', 'SAMUNDRA DAHAL', 'Absent'], ['KCE074BCT039', 'SANGAT ROKAYA', 'Absent'], ['KCE074BCT040', 'SHREE KRISHNA TUITUI', 'Absent'], ['KCE074BCT041', 'SHREEJAN KISEE', 'Absent'], ['KCE074BCT042', 'SUBIN TIMILSINA', 'Absent'], ['KCE074BCT043', 'SUJAN ACHARYA', 'Absent'], ['KCE074BCT044', 'SUJATA SUWAL', 'Absent'], ['KCE074BCT045', 'SUNIL BANMALA', 'Absent'], ['KCE074BCT046', 'SURAJ GOSAI', 'Absent'], ['KCE074BCT047', 'SUSHAN SHRESTHA', 'Absent'], ['KCE074BCT048', 'SUSHIL DYOPALA', 'Absent']]
subject_selected="Information System"
attendance_data = json.load(open(maindir+"\\Notebook_Scripts_Data\\data.json"))

p_list=[]
a_list=[]
for data in data:
    if data[2]=='Present' :
        p_list.append(data[0])
        attendance_data['student'][data[0]][subject_selected]['Present']+=1
    else:
        a_list.append(data[0])
    attendance_data['student'][data[0]][subject_selected]['total']+=1
temp={'date':datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
"present_list":p_list,
"absent_list":a_list}
attendance_data['attendance'][subject_selected].append(temp)
with open(maindir+"\\Notebook_Scripts_Data\\data.json","w" )as f:
    f.write(json.dumps(attendance_data)) 