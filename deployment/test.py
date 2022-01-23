import os

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
print(basedir[:-11])

context={}

if(context):
    print("asd")
else:
    print("no")