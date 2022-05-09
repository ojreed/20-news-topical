'''
author : abhayparashar31
This Program Take Height(CM) and Weight(KG) and Returns The BMI(Body Mass Index) Value As Output In a Pop up Box.
'''
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
import data_manager_2 as dm 
def get_alpha():
    alpha = float(ENTRY1.get())
    return alpha
def get_beta():
    beta = float(ENTRY2.get())
    return beta
def get_iter():
    num_iter = int(ENTRY3.get())
    return num_iter
def get_K():
    K = int(ENTRY4.get())
    return K
def get_size():
    size = int(ENTRY5.get())
    return size
def get_name():
    name = ENTRY6.get()
    return name

def check_all():
    entryList = [ENTRY1,ENTRY2,ENTRY3,ENTRY4,ENTRY5,ENTRY6]
    for entry in entryList:
        if entry.get() == "":
            return False
    return True


def browseFiles():
    numInpDirs = int(simpledialog.askstring(title = "Number of Training Directories", prompt = "Input the desired number of training directories"))
    dirNames = []
    for x in range(numInpDirs):
        dirNames.append(filedialog.askdirectory(title="add directory of training data"))
    return dirNames
    
    


def run_LDA():
    
    if (check_all()):
        dirNames = browseFiles()
        LDA = dm.LDAManager(get_name(),saftey = False)
        messagebox.showinfo("Result", "Begining New LDA Experiemnt")
        # LDA.defaultSetDataGroup()
        LDA.setDataGroup(dirNames)
        LDA.main(alpha=get_alpha(),beta=get_beta(),num_iter=get_iter(),K=get_K(),toy_size=get_size())
        LDA.visualize_topics()
        LDA.visualize_words(40)
        LDA.save()
        label_results()
    else:
        label_results()
        messagebox.showinfo("Result", "Missing Input, Try Again")


def label_results():
    TWO = Tk()
    TWO.geometry("600x400")
    TWO.configure(background="#68ace5")
    TWO.title("LDA Wizard - Label Topics")


    TWO.mainloop()

if __name__ == '__main__':


    TOP = Tk()
    TOP.geometry("400x600")
    TOP.configure(background="#68ace5")
    TOP.title("LDA Wizard")
    TOP.resizable(width=False, height=False)
    LABLE = Label(TOP, bg="#68ace5",fg="#ffffff", text="Welcome to Easy LDA", font=("Helvetica", 15, "bold"), pady=10)
    LABLE.place(x=40, y=0)
    LABLE1 = Label(TOP, bg="#ffffff", text="Enter Alpha:", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)
    LABLE1.place(x=40, y=60)
    ENTRY1 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY1.place(x=240, y=60)
    LABLE2 = Label(TOP, bg="#ffffff", text="Enter Beta:", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)

    LABLE2.place(x=40, y=121)
    ENTRY2 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY2.place(x=240, y=121)

    LABLE3 = Label(TOP, bg="#ffffff", text="Enter Iteration Count:", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)

    LABLE3.place(x=40, y=182)
    ENTRY3 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY3.place(x=240, y=182)

    LABLE4 = Label(TOP, bg="#ffffff", text="Desired Number of Topics:", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)

    LABLE4.place(x=40, y=243)
    ENTRY4 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY4.place(x=240, y=243)

    LABLE5 = Label(TOP, bg="#ffffff", text="Input Document Limit:", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)

    LABLE5.place(x=40, y=304)
    ENTRY5 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY5.place(x=240, y=304)

    LABLE6 = Label(TOP, bg="#ffffff", text="File Save Name:", bd=6,
                   font=("Helvetica", 10, "bold"), pady=5)

    LABLE6.place(x=40, y=365)
    ENTRY6 = Entry(TOP, bd=8, width=10, font="Roboto 11")
    ENTRY6.place(x=240, y=365)



    maxVal=365


    BUTTON = Button(bg="#000000",fg='#ffffff', bd=12, text="RUN", padx=33, pady=10, command=run_LDA,
                    font=("Helvetica", 20, "bold"))
    BUTTON.grid(row=5, column=0, sticky=W)
    BUTTON.place(x=115, y=maxVal+129)

    TOP.mainloop()