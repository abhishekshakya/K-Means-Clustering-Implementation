import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=500,n_features=2,centers=5,random_state = 3)

class cluster:
	def __init__(self,center,points):
		self.center = center
		self.points = points


class kmeans:

	def __init__(self,X,k=3,itration=5):

		mini = min([np.min(X[:,m]) for m in range(X.shape[1])])
		maxi = max([np.max(X[:,m]) for m in range(X.shape[1])])
		maxi = max(mini*-1,maxi)

		self.k = k
		self.X = X
		self.clusters = []
		for ith_cluster in range(k):
			center = maxi*((2*np.random.random(X.shape[1])) - 1)
			self.clusters.append(cluster(center,[]))

		for i in range(itration):
			self.assign_clusters()
			self.plot()
			self.update_clusters()
			

	def distance(self,x1,x2):
		return np.sqrt(np.sum((x1-x2)**2))

	def assign_clusters(self):
		for i in range(self.X.shape[0]):
			dist = []
			data = X[i]
			for ith_cluster in self.clusters:
				center = ith_cluster.center
				d = self.distance(center,data)
				dist.append(d)

			no = np.argmin(dist)
			self.clusters[no].points.append(data)
			


	def update_clusters(self):
		for ith_cluster in self.clusters:
			assigned_points = ith_cluster.points
			if len(assigned_points) > 0:
				new_center = np.mean(assigned_points,axis=0)
				ith_cluster.center = new_center
				ith_cluster.points = []

	def plot(self):
		if X.shape[1] > 2:
			print("plotting not possible")
		else:
			color = ['red','blue','orange','green','yellow']
			i = 0;
			for ith_cluster in self.clusters:
				pts = np.array(ith_cluster.points)
				print(pts.shape)
				try:
					plt.scatter(pts[:,0],pts[:,1],color=color[i])
				except:
					pass
				i +=1
				plt.scatter(ith_cluster.center[0],ith_cluster.center[1],color='black',marker='*')

			plt.show()




obj = kmeans(X,k=5,itration=5)

