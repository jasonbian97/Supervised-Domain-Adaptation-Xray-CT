'''
This code finds all images of patients of a specified VIRUS and X-Ray view and stores selected image to an OUTPUT directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in output folder
Code can be modified for any combination of selection of images
'''

import pandas as pd
import shutil
import os

import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory

# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
virus = "COVID-19" # Virus to look for
# x_ray_view = "PA" # View of X-Ray
modality = "X-ray"

metadata = "../raw/COVID-Xray/metadata.csv" # Meta info
imageDir = "../raw/COVID-Xray/images" # Directory of images
outputDir1 = '../cache/COVID-Xray_COVID19Xray' # Output directory to store selected images
outputDir2 = '../cache/COVID-Xray_NonCOVID19Xray'

if not os.path.exists(outputDir1):
	os.makedirs(outputDir1)
if not os.path.exists(outputDir2):
	os.makedirs(outputDir2)

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	if row["modality"] == modality:
		if row["finding"] == virus:
			filename = row["filename"].split(os.path.sep)[-1]
			filePath = os.path.sep.join([imageDir, filename])
			shutil.copy2(filePath, outputDir1)
		elif row["finding"] != "No Finding":
			filename = row["filename"].split(os.path.sep)[-1]
			filePath = os.path.sep.join([imageDir, filename])
			shutil.copy2(filePath, outputDir2)
		else:
			continue



