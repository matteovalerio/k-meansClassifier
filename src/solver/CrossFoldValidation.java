package solver;

import entity.Cluster;
import javafx.util.Pair;
import util.Utils;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class CrossFoldValidation {
    private int nFolds;
    private Instances source;
    private StringBuilder statistics = new StringBuilder();

    public CrossFoldValidation(int nFolds, Instances source) {
        this.nFolds = nFolds;
        this.source = source;
    }

    public String getStatistics() {
        return statistics.toString();
    }

    public void crossValidateModel(int percentage, int nClass) throws Exception {
        int elements = source.numInstances() / nFolds;
        int start = 0;
        int end = elements - 1;
        ArrayList<Double> accuracies = new ArrayList<>();
        source.randomize(new Random(1));
        System.out.println("=== " + percentage + "% ===");
        for (int i = 0; i < nFolds; i++) {
            System.out.println("Iteration n." + (i + 1));

            // Setup train and test set
            Instances trainingSet = new Instances(source);
            Instances testSet = new Instances(trainingSet, start, (end - start));
            testSet.setClassIndex(testSet.numAttributes() - 1);

            for (int j = start; j < trainingSet.numInstances(); j++) {
                trainingSet.delete(j);
            }

            Pair<Instances, Instances> dividedTrainingSet = Utils.extractInstancePercentage(percentage, trainingSet);
            trainingSet = new Instances(dividedTrainingSet.getValue());
            Instances extractedTrainingSet = new Instances(dividedTrainingSet.getKey());
            extractedTrainingSet.setClassIndex(extractedTrainingSet.numAttributes() - 1);
            ClusterGenerator s = new ClusterGenerator(nClass, trainingSet);
            s.classifyWithClusters(extractedTrainingSet);
          /*  for (Cluster c : s.getClusters()) {
                c.printInstances();
            }*/
            ClusterEvaluator evaluator = new ClusterEvaluator(testSet);
            evaluator.evaluateClassifier(s);
            accuracies.add(evaluator.getPercentageCorrect());

            start = end + 1;
            end += elements;
            if (i == nFolds - 1) {
                end = source.numInstances();
            }
            statistics.append(evaluator.getStatistics());
        } // end k-cross fold validation
        statistics.append("Accuracy ").append(percentage).append("%: ").append(Utils.getAveragePercentage(accuracies));
        System.out.println("Accuracy " + percentage + "%: " + Utils.getAveragePercentage(accuracies));
        System.out.println();
    }

}
