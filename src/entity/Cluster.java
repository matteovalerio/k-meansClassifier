package entity;

import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.UUID;

public class Cluster {
    private final UUID id = UUID.randomUUID();
    private double classLabel;
    private Instances instances;

    public Cluster(Instances instances) {
        this.instances = instances;
    }

    public String getId() {
        return id.toString();
    }

    public void setClassLabel() {
        HashMap<Double, Integer> counterClassIndexes = new HashMap<>();
        for (Instance i :
                instances) {
            double j = i.classValue();
            if (counterClassIndexes.containsKey(j)) {
                counterClassIndexes.put(j, counterClassIndexes.get(j) + 1);
            } else {
                counterClassIndexes.put(j, 1);
            }
        }
        this.classLabel = findMaxIndex(counterClassIndexes);
    }

    public double getClassLabel() {
        return classLabel;
    }

    public void insertInstance(Instance i) {
        instances.add(i);
    }

    public Instances getInstances() {
        return instances;
    }

    public int getNumInstances() {
        return instances.size();
    }

    private double findMaxIndex(HashMap<Double, Integer> map) {
        Double[] keys = map.keySet().toArray(new Double[0]);
        int maxVal = -1;
        double maxKey = -1;
        for (Double key : keys) {
            int currentVal = map.get(key);
            if (maxVal < currentVal) {
                maxVal = currentVal;
                maxKey = key;
            }
        }
        return maxKey;
    }

    public void printInstances() {
        System.out.println("=== Cluster n." + (classLabel) + ", instances: ===");
        for (Instance i :
                instances) {
            System.out.println(i);
        }
    }
}