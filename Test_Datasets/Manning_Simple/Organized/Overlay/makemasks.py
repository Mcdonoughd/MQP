import os

for i in range(1,19):
	parentfolder = "./img"+str(i)
	if not os.path.isdir(parentfolder):
		os.mkdir(parentfolder)
		os.mkdir(os.path.join(parentfolder,"Masks"))
