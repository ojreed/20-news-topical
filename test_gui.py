'''
author : abhayparashar31
This Program Take Height(CM) and Weight(KG) and Returns The BMI(Body Mass Index) Value As Output In a Pop up Box.
'''
from tkinter import *
from tkinter import messagebox
def get_height():
    height = float(ENTRY2.get())
    return height
def get_weight():
    weight = float(ENTRY1.get())
    return weight
def calculate_bmi(): 
    try:
        height = get_height()
        weight = get_weight()
        height = height / 100.0
        bmi = weight / (height ** 2)
    except ZeroDivisionError:
        messagebox.showinfo("Result", "Please enter positive height!!")
    except ValueError:
        messagebox.showinfo("Result", "Please enter valid data!")
    else:
        messagebox.showinfo("Your BMI Calculated is : ", bmi)

if __name__ == '__main__':
    TOP = Tk()
    TOP.bind("<Return>", calculate_bmi)
    TOP.geometry("400x400")
    TOP.configure(background="#8c52ff")
    TOP.title("BMI Calculator")
    TOP.resizable(width=False, height=False)
    LABLE = Label(TOP, bg="#8c52ff",fg="#ffffff", text="Welcome to BMI Calculator", font=("Helvetica", 15, "bold"), pady=10)
    LABLE.place(x=55, y=0)
    LABLE1 = Label(TOP, bg="#ffffff", text="Enter Weight (in kg):", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)
    LABLE1.place(x=55, y=60)
    ENTRY1 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY1.place(x=240, y=60)
    LABLE2 = Label(TOP, bg="#ffffff", text="Enter Height (in cm):", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)
    LABLE2.place(x=55, y=121)
    ENTRY2 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY2.place(x=240, y=121)
    BUTTON = Button(bg="#000000",fg='#ffffff', bd=12, text="BMI", padx=33, pady=10, command=calculate_bmi,
                    font=("Helvetica", 20, "bold"))
    BUTTON.grid(row=5, column=0, sticky=W)
    BUTTON.place(x=115, y=250)
    TOP.mainloop()