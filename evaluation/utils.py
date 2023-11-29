import csv
import os

header = ["input", "cham", "cham_base", "iou", "iou_base", "fscore", "fscore_base"]

def create_csv(filename):
  with open(f"{filename}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

def clean_csv(filename):
  with open(f"{filename}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
      
def add_csv(filename, values):
  if not os.path.isfile(f"{filename}.csv") :
    create_csv(filename)

  with open(f"{filename}.csv", mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(values)