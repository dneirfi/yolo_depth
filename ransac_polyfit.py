# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys


def ransac_polyfit(x, y, order=3, n=20, k=100, t=0.1, d=100, f=0.8):
  # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
  
  # n – minimum number of data points required to fit the model
  # k – maximum number of iterations allowed in the algorithm
  # t – threshold value to determine when a data point fits a model
  # d – number of close data points required to assert that a model fits well to data
  # f – fraction of close data points required
    x = np.asarray(x);	y = np.asarray(y); 
    besterr = np.inf
    bestfit = None
    #for kk in xrange(k):
    #maybeinliers=[]
    print('# of points : ' + str(len(x)))
    for kk in range(0, k):
        maybeinliers = np.random.randint(len(x), size=n)
        #print('type(maybeinliers) before tolist: ');  print(type(maybeinliers));
        #maybeinliers = maybeinliers.tolist()

        #print('type(maybeinliers) after tolist: ');  print(type(maybeinliers));
        #print('type(x) : ');  print(type(x));
        print('maybeinliers : ' + str(maybeinliers))
        print('x[maybeinliers] : ');    print(x[maybeinliers])
        print('y[maybeinliers] : ');    print(y[maybeinliers])
        print(str(kk) + ' / ' + str(k))
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
        print('alsoinliers : ');    print(alsoinliers);
        print('sum(alsoinliers) : ' + str(sum(alsoinliers)))
        print('d : ' + str(d))
        print('len(x) * f : ' + str(len(x)*f))
        if sum(alsoinliers) > d and sum(alsoinliers)> len(x)*f:
            print('222');
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            print('333');
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
            print('444');
            if thiserr < besterr:
		#print('555');
                bestfit = bettermodel
                print('besterr is changed from : ' + str(besterr) + ' to ' + str(thiserr) + ' at ' + str(kk) + ' / ' + str(k))
                besterr = thiserr
    return bestfit


#t1 = [1, 2, 3, 4, 5]
#t2 = t1 > 3
#print(t2)
filename = sys.argv[1]
f1 = open (filename,'r')

lines =f1.readlines()

millimeter=[]
disparity=[]
distance=[]
for line in lines:
	if line[0]=="#":
		continue	
	item = line.split(" ")
	millimeter.append(float(item[1]))
	disparity.append(float(item[2]))
	
	i = item[0].split("_")
	distance.append(float(i[1]))
f1.close()


#print(millimeter)
print(disparity)
print(distance)

#ransac_polyfit(disparity, distance, order=3, n=20, k=100, t=0.1, d=100, f=0.8)
bestfit_disparity = ransac_polyfit(disparity, distance, order=3, n=7, k=200, t=15.0, d=5, f=0.5)

print('bestfit_disparity : ');	print(bestfit_disparity)
bestfit_millimeter = ransac_polyfit(millimeter, distance, order=3, n=7, k=200, t=15.0, d=5, f=0.5)

print('bestfit_millimeter : ');	print(bestfit_millimeter)



bestfit_disparity = bestfit_disparity.tolist()
bestfit_millimeter = bestfit_millimeter.tolist()
#fn_res = sys.argv[2]
with open (sys.argv[2],'w') as outfile:
    outfile.write(" ".join(str(coef) for coef in bestfit_disparity[::-1]) + '\n')
    outfile.write(" ".join(str(coef) for coef in bestfit_millimeter[::-1]) + '\n')

#x2=xrange(500,3000)
x2=range(300, 2500)
#x1=xrange(0,100)
x1=range(10, 120)
# bestfit_disparity[0] : 3차항
# bestfit_disparity[1] : 2차항
#,bestfit_disparity[2] : 1차항
# bestfit_disparity[3] : 상수항
p1=np.poly1d([bestfit_disparity[0],bestfit_disparity[1],bestfit_disparity[2], bestfit_disparity[3]])
p2=np.poly1d([bestfit_millimeter[0],bestfit_millimeter[1],bestfit_millimeter[2],bestfit_millimeter[3]])
#print(bestfit_disparity[0],bestfit_disparity[1],bestfit_disparity[2],bestfit_disparity[3])
#print(bestfit_millimeter[0],bestfit_millimeter[1],bestfit_millimeter[2],bestfit_millimeter[3])
y1=p1(x1)
y2=p2(x2)
plt.subplot(2,1,1)
plt.plot(x1, y1, label = 'disparity')
plt.scatter(disparity, distance)
plt.ylim([0,2000])

plt.subplot(2,1,2)
plt.plot(x2, y2, label = 'millimeter')
plt.scatter(millimeter, distance)
plt.ylim([0,2000])

plt.show()

#y1=[]
#y2=[]


#for t in xlen:
#    y1.append(bestfit_disparity[0]*t**3 + bestfit_disparity[1]*t**2 + bestfit_disparity[2]*t + bestfit_disparity[3])
#    y2.append(bestfit_millimeter[0]*t**3 + bestfit_millimeter[1]*t**2 + bestfit_millimeter[2]*t + bestfit_millimeter[3])
#for i in xlen:
#    y2.append(a2*i**3 + b2*i**2 + c2*i + d2)

#plt.plot(xlen, y1, label = 'disparity')
#plt.plot(xlen, y2, label = 'millimeter')
#plt.legend(loc='upper right')

