import solver.CrossFoldValidation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.FileWriter;

public class MainClass {
    public static void main(String[] args) {
        try {
            Instances source = DataSource.read("data/iris.arff");
            int nFolds = 10;
            int nClass = 3;
            int[] percentages = {30, 20, 10, 5};
            File file = new File("data/results1.txt");
            FileWriter writer = new FileWriter(file);
            writer.write("=== IRIS DATA RESULTS ===\n");
            for (int percentage : percentages) {
                CrossFoldValidation validation = new CrossFoldValidation(nFolds, source);
                validation.crossValidateModel(percentage, nClass);
                writer.write(validation.getStatistics());
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}