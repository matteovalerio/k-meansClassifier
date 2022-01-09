package solver;

import entity.Cluster;
import weka.clusterers.SimpleKMeans;
import weka.core.*;

import java.util.ArrayList;

public class ClusterGenerator {

    private final int numClass;
    private final Instances instances;
    private final ArrayList<Cluster> clusters;
    private final SimpleKMeans simpleKMeans;
    private NormalizableDistance distanceFunction = new EuclideanDistance();

    /**
     * Since we are using the Simple k-means, if not provided the distance function is the euclidean distance by default
     */
    public ClusterGenerator(int numClass, Instances instances) throws Exception {
        checkParameters(numClass, instances);
        this.numClass = numClass;
        this.instances = instances;
        clusters = new ArrayList<>(numClass);
        simpleKMeans = generateClusterModel();
    }

    public ClusterGenerator(int numClass, Instances instances, NormalizableDistance distanceFunction) throws Exception {
        checkParameters(numClass, instances);
        this.numClass = numClass;
        this.instances = instances;
        this.distanceFunction = distanceFunction;
        clusters = new ArrayList<>(numClass);
        simpleKMeans = generateClusterModel();
    }

    private void checkParameters(int numClass, Instances instances) throws Exception {
        if (numClass <= 0) {
            throw new Exception("Number of cluster must be >= 0");
        }
        if (instances.numInstances() == 0) {
            throw new Exception("Number of instances must be > 0");
        }
    }

    public Cluster getCluster(int index) {
        return clusters.get(index);
    }

    public ArrayList<Cluster> getClusters() {
        return clusters;
    }

    public int getNumClass() {
        return numClass;
    }

    public SimpleKMeans getSimpleKMeans() {
        return simpleKMeans;
    }

    /**
     * Generate cluster model and create clusters
     */
    private SimpleKMeans generateClusterModel() throws Exception {
        SimpleKMeans simpleKMeans = new SimpleKMeans();
        simpleKMeans.setNumClusters(numClass);
        simpleKMeans.setDistanceFunction(distanceFunction);
        simpleKMeans.buildClusterer(instances);
        return simpleKMeans;
    }

    /**
     * It classifies new data using the distance from the centroids of the clusters generated.
     */
    public ArrayList<Cluster> classifyWithClusters(Instances data) throws Exception {
        Instances centroids = simpleKMeans.getClusterCentroids();

        // Initialize clusters with only centroids
        for (int i = 0; i < centroids.numInstances(); i++) {
            Instances tmp = new Instances(data);
            tmp.delete();
            tmp.add(centroids.get(i));
            clusters.add(new Cluster(tmp));
        }

        // for each new data, classify to the correct cluster
        for (Instance instance : data) {
            classifyNewInstance(instance);
        }

        // generate labels for clusters
        generateClassLabels();
        return clusters;
    }


    /**
     * Given a new instance, classify it to the correct cluster
     */
    public int classifyNewInstance(Instance instance) throws Exception {
        Instances centroids = simpleKMeans.getClusterCentroids();
        DistanceFunction distanceFunction = simpleKMeans.getDistanceFunction();
        int minIndexValue = -1;
        for (int j = 0; j < centroids.numInstances(); j++) {
            if (minIndexValue == -1 ||
                    (distanceFunction.distance(instance, centroids.instance(j)) <
                            distanceFunction.distance(instance, centroids.instance(minIndexValue)))) {
                minIndexValue = j;
            }
        }
        clusters.get(minIndexValue).insertInstance(instance);
        return minIndexValue;
    }

    public void generateClassLabels() {
        for (Cluster cluster : clusters) {
            cluster.setClassLabel();
        }
    }

}