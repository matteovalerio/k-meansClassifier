package util;

import javafx.util.Pair;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Utils {
    public static final float PERCENT = 100;

    public static Pair<Instances, Instances> extractInstancePercentage(int percentage, Instances instances) {
        Instances extractedInstances = new Instances(instances);
        extractedInstances.delete();
        int nExtractionInstances = Math.round((percentage / PERCENT) * instances.numInstances());
        for (int i = 0; i < nExtractionInstances; i++) {
            int r = ThreadLocalRandom.current().nextInt(0, instances.numInstances() - 1);
            extractedInstances.add(instances.instance(r));
            instances.delete(r);
        }
        return new Pair<>(extractedInstances, instances);
    }

    public static double getAveragePercentage(ArrayList<Double> list) {
        double tot = 0.0;
        for (Double aDouble : list) {
            tot += aDouble;
        }
        return tot / list.size();
    }
}
