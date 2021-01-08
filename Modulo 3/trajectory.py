import csv
with open("map_3.txt") as file:
    reader = csv.reader(file)
    f = open("trajectory_3.txt" , "a")
    for row in reader:
        #print(str(row[0]) + " " + str(row[1]) + "\n")
        f.write(str(row[0]) + "," + str(row[1]) + "\n")
f.close()
