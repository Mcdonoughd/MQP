import os
import subprocess
import sys
import pymatlab
import shutil
from tkinter import *
from tkinter import Menu
from tkinter.filedialog import askdirectory

# Todo set up preporcess button or have it done automatically when loaded dataset
# here we should set up pre-processing to set up the file format for the channels
# database -> csv format to read and store cells?

# Cell DB
# cell id, time point, centroid (x,y), cell foci count, green channel ref, red channel ref, mask ref,





# todo Functionallity
# run cell tracking
# run Foci tracking

# todo:
#  be able to read look at a cell with different channels and masks




def LoadDataset():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askdirectory()  # show an "Open" dialog box and return the path to the selected file
    project_name = filename.split("/")[-1]

    shutil.copytree(filename,"./CIRA/Projects/"+project_name)
    print("Loaded Project: "+project_name)
    window.title("CIRA: "+project_name)

def NewProject():
    print("NEW PROJECT")


def OpenProject():
    print("Open PROJECT")


### GUI System ###
os.chdir(os.getcwd())
print("Working in: "+os.getcwd())
# if path does not exist
if not os.path.exists("./CIRA") or not os.path.exists("./CIRA/Projects"):
    print("Could not find CIRA Projects Folder...")
    os.makedirs("./CIRA/Projects")



window = Tk()

window.title("CIRA")

menu = Menu(window)

new_item = Menu(menu)

new_item.add_command(label='Load Dataset',command=LoadDataset)


new_item.add_command(label='Edit')

menu.add_cascade(label='File', menu=new_item)

window.config(menu=menu)

window.mainloop()
