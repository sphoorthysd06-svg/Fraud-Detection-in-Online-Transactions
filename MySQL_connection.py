# ============================================================
# MYSQL CONNECTION USING PYMYSQL
# FRAUD DETECTION IN ONLINE TRANSACTIONS
# ============================================================

import pymysql

print("PyMySQL imported successfully.")

# CONNECT TO MYSQL DATABASE

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="system",
    database="fraud_detection_db"
)

print("Connected to MySQL successfully!")

# CREATE CURSOR

cursor = conn.cursor()

# CHECK TOTAL RECORDS

cursor.execute("SELECT COUNT(*) FROM fraud_transactions")

result = cursor.fetchone()

print("Total records in MySQL:", result[0])

# CLOSE CONNECTION

cursor.close()
conn.close()

print("MySQL connection closed.")