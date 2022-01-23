import mysql.connector

mydb = mysql.connector.connect(
    host="sql6.freesqldatabase.com",
    user="sql6467504",
    passwd="DyECurrXmg",
    database="sql6467504"
)

# making cursor
cursor = mydb.cursor(dictionary=True)
username="KCE074BCT023"
password="b532f1b8869ba150ee239776740ccb7a7c6040a59d14e6c227ca48c5bf3a3315"
cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))

account = cursor.fetchone()
print(account)