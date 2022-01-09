package solver;

import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileWriter;

import static util.Utils.PERCENT;

public class ClusterEvaluator {
    private final Instances testSet;
    private int nCorrect = 0;
    private int nIncorrect = 0;
    private int tot = 0;
    private double percentageCorrect = 0.0;
    private double percentageIncorrect = 0.0;
    private final StringBuilder st = new StringBuilder();

    public ClusterEvaluator(Instances testSet) {
        this.testSet = new Instances(testSet);
    }

    public String getStatistics() {
        return st.toString();
    }


    public double getPercentageCorrect() {
        return percentageCorrect;
    }

    public double getPercentageIncorrect() {
        return percentageIncorrect;
    }

    public int totalElementsProcessed() {
        return tot;
    }

    public int getCorrectElements() {
        return nCorrect;
    }

    public int getIncorrectElements() {
        return nIncorrect;
    }

    public void evaluateClassifier(ClusterGenerator clusterGenerator) throws Exception {
        int[] nClasses = new int[clusterGenerator.getNumClass()];
        int[] truePositives = new int[clusterGenerator.getNumClass()];
        int[] falsePositives = new int[clusterGenerator.getNumClass()];
        int[] correctDistribution  = new int[clusterGenerator.getNumClass()];
        for (Instance i : testSet) {
            int index = clusterGenerator.classifyNewInstance(i);
            double classLabel = clusterGenerator.getCluster(index).getClassLabel();
            nClasses[(int)classLabel] += 1;
            correctDistribution[(int)i.classValue()] += 1;
            System.out.println("Instance " + i + " classified as " + classLabel);
            if (classLabel != i.classValue()) {
                nIncorrect++;
                falsePositives[(int) classLabel] += 1;
            } else {
                nCorrect++;
                truePositives[(int) classLabel] += 1;
            }
        }
        tot = testSet.numInstances();
        percentageCorrect = (nCorrect * PERCENT) / tot;
        percentageIncorrect = (nIncorrect * PERCENT) / tot;
        st.append("\nTEST SET RESULTS:\nN°correct: " + nCorrect + "\nN°incorrect: " + nIncorrect + "\nTot: " + tot + "\nPercentage correct: " + percentageCorrect + " %" + "\nPercentage incorrect: " + percentageIncorrect + " %\n\n");

        System.out.println("\nTEST SET RESULTS:\n");
        System.out.println("N°correct: " + nCorrect);
        System.out.println("N°incorrect: " + nIncorrect);
        System.out.println("Tot: " + tot);
        System.out.println("Percentage correct: " + percentageCorrect + " %");
        System.out.println("Percentage incorrect: " + percentageIncorrect + " %\n\n");

        for (int i = 0; i < nClasses.length; i++) {
            int trueNegative = tot - (nClasses[i] - correctDistribution[i]);
            int truePositive = truePositives[i];
            int falsePositive = (tot - correctDistribution[i]) - (nClasses[i] - truePositive);
            int falseNegative = correctDistribution[i] - truePositive;
            st.append("\nClass ").append(i);
            st.append("\nTPR: ").append(tpr(truePositive, falseNegative));
            st.append("\nFPR: ").append(fpr(falsePositive, trueNegative)).append("\n");
            System.out.println("TP-" + truePositive + "-FN-" + falseNegative + "-FP-" + falsePositive + "-TN-" + trueNegative + "-class-" + nClasses[i]);
            System.out.println("TPR class " + i + ": " + tpr(truePositive, falseNegative));
            System.out.println("FPR: class " + i + ": " + fpr(falsePositive, falseNegative));
        }
    }

    private double fpr(int falsePositive, int trueNegative) {
        int tot = falsePositive + trueNegative;
        if (tot == 0)
            return 0;
        return (double) falsePositive / (falsePositive + trueNegative);
    }

    private double tpr(int truePositive, int falseNegative) {
        int tot = truePositive + falseNegative;
        if (tot == 0)
            return 1;
        return (double) truePositive / (truePositive + falseNegative);
    }
}
