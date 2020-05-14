from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def iris_kmean():
    iris = load_iris()
    x = iris.data
    print(iris.keys())
    clusters = []
    for i in range(1,10):
        km = KMeans(n_clusters= i).fit(x)
        clusters.append(km.inertia_)

    plt.plot(clusters)
    plt.xlabel('Number of clusters')
    plt.ylabel('Total square distance')
    plt.show()

if __name__ == "__main__":
    iris_kmean()

#Elbow heuristic confirms the ideal number of clusters is the same as the assumed number of clusters in Iris dataset i.e. three.