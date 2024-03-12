import os

os.chdir("Data.nosync/All data in one place")

for oldfilename in os.listdir():
    if oldfilename[0] == ' ':
        newfilename = 'lambda' + oldfilename
        os.rename(oldfilename, newfilename)
