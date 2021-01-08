import csv
import matplotlib.pyplot as plt

X = []
Y = []
with open("trajectory_3.txt") as file:
    next(file)
    reader = csv.reader(file)
    for row in reader:
        X.append(float(row[0]))
        Y.append(float(row[1]))

#Map 1
# start_x = 234.269928
# start_y = 55.482471
# end_x = 334.734589
# end_y = 273.742859
#Map 2
# start_x = 189.670715
# start_y = 293.537994
# end_x = -7.415379
# end_y = 142.190155
#Mao 3
start_x = -55.049988
start_y = 0.599630
end_x = -5.532081
end_y = -79.032463

plt.plot(start_x, start_y,'+r',markersize=12, linewidth = 8)
plt.plot(end_x, end_y,'+b',markersize=12, linewidth = 8)
plt.text(start_x,start_y, 'Start point')
plt.text(end_x,end_y, 'End point')

for i in range(len(X)):
    #print("X: " + str(X[i]) + " Y: " + str(Y[i]))
    plt.plot(X[i],Y[i],'ok',markersize=4)


plt.xlim(-90,35)
plt.ylim(-210, 40)
plt.title("Map 3 Route")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
# print("X: " + str(X[0]) + " Y: " + str(Y[0]))
# print("X: " + str(X[1]) + " Y: " + str(Y[1]))
# print("X: " + str(X[2]) + " Y: " + str(Y[2]))
# print("X: " + str(X[3]) + " Y: " + str(Y[3]))
# print("X: " + str(X[4]) + " Y: " + str(Y[4]))
# print("X: " + str(X[5]) + " Y: " + str(Y[5]))
# print("X: " + str(X[6]) + " Y: " + str(Y[6]))
# print("X: " + str(X[7]) + " Y: " + str(Y[7]))
# print("X: " + str(X[8]) + " Y: " + str(Y[8]))
# print("X: " + str(X[9]) + " Y: " + str(Y[9]))
# print("Xmin: " + str(min(X)))
# print("Xmax: " + str(max(X)))
# print("Ymin: " + str(min(Y)))
# print("Ymax: " + str(max(Y)))
#Map1
#Xmin: 158.020432
#Xmax: 334.895020
#Ymin: 1.956516
#Ymax: 271.158722
#Map2
#Xmin: -7.409455
#Xmax: 193.73999
#Ymin: 105.390747
#Ymax: 302.559875
#Map3
#Xmin: -74.539406
#Xmax: 19.44516
#Ymin: -195.275085
#Ymax: 23.763405


plt.show()
