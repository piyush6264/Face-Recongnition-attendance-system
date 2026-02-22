import customtkinter as ctk   # modern Tkinter
import tkinter as tk
from tkinter import ttk, messagebox as mess
import cv2
import os
import csv
import numpy as np
import pandas as pd
import time
from PIL import Image

# ---------------- Utility Functions ----------------
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_student_data():
    """Load student data from master CSV file into dictionary."""
    global student_data
    try:
        df = pd.read_csv("Copy-of-mca-students-24-26-_1__1_.csv")
        student_data = dict(zip(df["UID"].astype(str), df["Name"]))
    except Exception as e:
        mess.showerror("Error", f"Unable to load student data: {e}")
        student_data = {}

def on_id_select(event=None):
    """Auto-fill student name when ID is selected."""
    selected_id = id_combo.get()
    if selected_id in student_data:
        txt2.delete(0, tk.END)
        txt2.insert(0, student_data[selected_id])

# ---------------- Validation ----------------
def validate_student_details(user_id, user_name):
    try:
        df = pd.read_csv("Copy-of-mca-students-24-26-_1__1_.csv")
        user_id = str(user_id)
        user_name = str(user_name).strip()
        
        student = df[df["UID"] == user_id]
        if not student.empty:
            if student.iloc[0]["Name"].strip() == user_name:
                return True
        return False
    except Exception as e:
        print(f"Error reading student CSV: {e}")
        return False

def check_student_exists(user_id, user_name):
    if not validate_student_details(user_id, user_name):
        mess.showerror("Invalid Student", 
                      "Student ID or Name not found in the master list. Please check the details.")
        return True
    
    if not os.path.exists("StudentDetails/StudentDetails.csv"):
        return False
    
    with open("StudentDetails/StudentDetails.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and len(row) >= 2:
                if row[0] == user_id or row[1].lower() == user_name.lower():
                    mess.showerror("Registration Error", 
                                 "This student is already registered in the system.")
                    return True
    return False

# ---------------- Image Capture ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def take_images():
    assure_path_exists("TrainingImage/")
    assure_path_exists("StudentDetails/")
    user_id = id_combo.get()
    user_name = txt2.get()
    
    if not user_id or not user_name:
        mess.showerror("Invalid Input", "Please select an ID and ensure name is filled.")
        return
    
    if check_student_exists(user_id, user_name):
        return
    
    cam = cv2.VideoCapture(0)
    count = 0
    while count < 10:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"TrainingImage/{user_name}.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Taking Images", img)
        cv2.waitKey(1)
    
    cam.release()
    cv2.destroyAllWindows()
    
    if count > 0:
        with open("StudentDetails/StudentDetails.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user_id, user_name])
        mess.showinfo("Success", f"{count} Images saved for ID: {user_id}, Name: {user_name}")
    else:
        mess.showwarning("No Face Detected", "No face detected. Try again.")

# ---------------- Training ----------------
def train_images():
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    
    id_mapping = {}
    current_numeric_id = 1
    
    for image_path in os.listdir("TrainingImage/"):
        try:
            img_path = os.path.join("TrainingImage/", image_path)
            img = Image.open(img_path).convert("L")
            img_np = np.array(img, "uint8")
            
            student_id = image_path.split(".")[1]
            if student_id not in id_mapping:
                id_mapping[student_id] = current_numeric_id
                current_numeric_id += 1
            
            numeric_id = id_mapping[student_id]
            faces.append(img_np)
            ids.append(numeric_id)
            
        except Exception as e:
            print(f"Skipping file {image_path}: {e}")
    
    if faces:
        with open("TrainingImageLabel/id_mapping.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["StudentID", "NumericID"])
            for student_id, numeric_id in id_mapping.items():
                writer.writerow([student_id, numeric_id])
        
        recognizer.train(faces, np.array(ids))
        recognizer.save("TrainingImageLabel/Trainer.yml")
        mess.showinfo("Success", "Model trained successfully!")
    else:
        mess.showerror("Error", "No images found for training.")

# ---------------- Attendance ----------------
def save_attendance(user_id, user_name, status="IN"):
    file_path = "Attendance.csv"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["ID", "Name", "Status", "Timestamp"])
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    
    if status == "IN":
        student_entries = df[(df["ID"] == user_id) & (df["Name"] == user_name)]
        if not student_entries.empty and student_entries.iloc[-1]["Status"] == "IN":
            mess.showwarning("Warning", f"{user_name} is already marked as IN!")
            return
    
    new_entry = pd.DataFrame([{
        "ID": user_id,
        "Name": user_name,
        "Status": status,
        "Timestamp": current_time
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_path, index=False)
    
    status_text = "entered" if status == "IN" else "exited"
    update_attendance_list(f"{user_name} {status_text} at {current_time}")

def mark_exit(user_id, user_name):
    file_path = "Attendance.csv"
    if not os.path.exists(file_path):
        mess.showwarning("Error", "No attendance records found.")
        return False

    df = pd.read_csv(file_path)
    student_entries = df[(df["ID"] == user_id) & (df["Name"] == user_name)]
    if student_entries.empty or student_entries.iloc[-1]["Status"] != "IN":
        mess.showwarning("Error", "No valid IN record found for exit.")
        return False

    last_in_time = pd.to_datetime(student_entries.iloc[-1]["Timestamp"])
    current_time = pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S"))
    duration = (current_time - last_in_time).total_seconds()

    if duration < 60:
        mess.showwarning("Exit Denied", "You cannot exit less than one minute after entry.")
        return False

    save_attendance(user_id, user_name, "OUT")
    return True

# ---------------- Recognition ----------------
def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("TrainingImageLabel/Trainer.yml")
    except:
        mess.showerror("Error", "No trained model found. Train first.")
        return
    
    id_mapping = {}
    try:
        with open("TrainingImageLabel/id_mapping.csv", "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                student_id, numeric_id = row
                id_mapping[int(numeric_id)] = student_id
    except:
        mess.showerror("Error", "ID mapping file not found. Please train the model again.")
        return
    
    names = {}
    with open("StudentDetails/StudentDetails.csv", "r") as file:
        for row in file:
            row = row.strip()
            if not row:
                continue
            values = row.split(",")
            if len(values) < 2:
                continue
            user_id, user_name = values[:2]
            names[user_id] = user_name.strip()
    
    attendance_data = pd.DataFrame(columns=["ID", "Name", "Status", "Timestamp"])
    if os.path.exists("Attendance.csv"):
        try:
            attendance_data = pd.read_csv("Attendance.csv")
        except:
            attendance_data = pd.DataFrame(columns=["ID", "Name", "Status", "Timestamp"])
    
    cam = cv2.VideoCapture(0)
    start_time = time.time()
    timeout = 5
    recognition_start_time = None
    recognized_face = None
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        elapsed_time = time.time() - start_time
        remaining_time = max(0, timeout - elapsed_time)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            numeric_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50 and numeric_id in id_mapping:
                student_id = id_mapping[numeric_id]
                name = names.get(student_id, "Unknown")
                
                student_entries = attendance_data[
                    (attendance_data["ID"] == student_id) & 
                    (attendance_data["Name"] == name)
                ]
                
                if student_entries.empty or student_entries.iloc[-1]["Status"] == "OUT":
                    status = "IN"
                    color = (0, 255, 0)
                else:
                    status = "OUT"
                    color = (0, 0, 255)
                
                cv2.putText(img, f"{name} ({status})", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                if recognition_start_time is None:
                    recognition_start_time = time.time()
                    recognized_face = (student_id, name, status)
                
                if time.time() - recognition_start_time >= 2:
                    if status == "IN":
                        save_attendance(student_id, name, "IN")
                        mess.showinfo("Attendance", f"Entry marked for {name}")
                    else:
                        if mark_exit(student_id, name):
                            mess.showinfo("Attendance", f"Exit marked for {name}")

                    cam.release()
                    cv2.destroyAllWindows()
                    return
            else:
                cv2.putText(img, "Unknown", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        timer_text = f"Time left: {int(remaining_time)}s"
        cv2.putText(img, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if recognition_start_time is not None:
            recognition_time_left = max(0, 2 - (time.time() - recognition_start_time))
            if recognition_time_left > 0:
                cv2.putText(img, f"Marking attendance in: {int(recognition_time_left)}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", img)
        key = cv2.waitKey(1)
        
        if remaining_time <= 0:
            cv2.putText(img, "Camera closing...", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Recognition", img)
            cv2.waitKey(1000)
            break
            
        if key == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

# ---------------- Attendance List ----------------
def update_attendance_list(name):
    attendance_list.insert(tk.END, f"{name}\n")
    attendance_list.see(tk.END)

def delete_attendance():
    file_path = "Attendance.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        attendance_list.delete(1.0, tk.END)
        mess.showinfo("Success", "Attendance records deleted.")
    else:
        mess.showwarning("Error", "No attendance records found to delete.")

# ---------------- Modern UI ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Face Recognition Attendance System")
window.geometry("900x650")

student_data = {}
load_student_data()

# Title
title_label = ctk.CTkLabel(window, text="🎓 Face Recognition Attendance System",
                           font=ctk.CTkFont(size=24, weight="bold"))
title_label.pack(pady=20)

# Frame for student input
frame_input = ctk.CTkFrame(window, corner_radius=15)
frame_input.pack(pady=10, padx=20, fill="x")

id_label = ctk.CTkLabel(frame_input, text="Select ID:", font=ctk.CTkFont(size=14))
id_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

id_combo = ttk.Combobox(frame_input, width=25, font=("Arial", 12), values=list(student_data.keys()))
id_combo.grid(row=0, column=1, padx=10, pady=10)
id_combo.bind('<<ComboboxSelected>>', on_id_select)

name_label = ctk.CTkLabel(frame_input, text="Name:", font=ctk.CTkFont(size=14))
name_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

txt2 = ctk.CTkEntry(frame_input, width=250, font=ctk.CTkFont(size=14))
txt2.grid(row=1, column=1, padx=10, pady=10)

instruction = ctk.CTkLabel(frame_input, text="ℹ️ Select ID to auto-fill name",
                           font=ctk.CTkFont(size=12, slant="italic"))
instruction.grid(row=2, column=0, columnspan=2, pady=(0, 10))

# Frame for buttons
frame_buttons = ctk.CTkFrame(window, corner_radius=15)
frame_buttons.pack(pady=20)

btn_take = ctk.CTkButton(frame_buttons, text="📷 Take Images", width=160, height=40,
                         fg_color="green", hover_color="#006400", command=take_images)
btn_take.grid(row=0, column=0, padx=15, pady=15)

btn_train = ctk.CTkButton(frame_buttons, text="🧠 Train Model", width=160, height=40,
                          fg_color="blue", hover_color="#00008B", command=train_images)
btn_train.grid(row=0, column=1, padx=15, pady=15)

btn_recognize = ctk.CTkButton(frame_buttons, text="👤 Recognize Face", width=160, height=40,
                              fg_color="orange", hover_color="#FF8C00", command=track_images)
btn_recognize.grid(row=0, column=2, padx=15, pady=15)

btn_delete = ctk.CTkButton(frame_buttons, text="🗑️ Delete Attendance", width=160, height=40,
                           fg_color="red", hover_color="#8B0000", command=delete_attendance)
btn_delete.grid(row=0, column=3, padx=15, pady=15)

# Attendance log frame
frame_log = ctk.CTkFrame(window, corner_radius=15)
frame_log.pack(pady=20, padx=20, fill="both", expand=True)

log_label = ctk.CTkLabel(frame_log, text="📑 Attendance Log",
                         font=ctk.CTkFont(size=18, weight="bold"))
log_label.pack(pady=10)

attendance_list = tk.Text(frame_log, height=12, width=70, font=("Consolas", 12),
                          bg="#1e1e1e", fg="white", insertbackground="white")
attendance_list.pack(pady=10, padx=10)

window.mainloop()
