import pandas as pd

df = pd.read_csv("Attendance.csv")
df.to_excel("Attendance.xlsx", index=False)