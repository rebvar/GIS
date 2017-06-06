/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package istgis;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author sehossei
 */
public class Benchmarks {

    public static void WCFolds(Instances testSet, int folds, String file, FileWriter fout, String name) throws Exception {

        ArrayList<FastVector> preds = new ArrayList();
        double vals[] = null;
        Instances tssCopy = new Instances(testSet);
        tssCopy.randomize(DPLIB.rnd);
        tssCopy.stratify(folds);
        for (int j = 0; j < folds; j++) {
            Instances cvtrain = new Instances(tssCopy.trainCV(folds, j));
            Instances cvtest = new Instances(tssCopy.testCV(folds, j));

            if (name.toLowerCase().contains("infogain")) {
                int indi[] = DPLIB.fSelectInfoGain(cvtrain);
                if (DPLIB.useIterativeInfoGainSubsetting) {
                    indi = DPLIB.iterativeInfoGainSubsetting(cvtrain, indi);
                } else {
                    indi = DPLIB.getTopX(indi);
                }
                cvtrain = DPLIB.fSelectSet(cvtrain, indi);
                cvtest = DPLIB.fSelectSet(cvtest, indi);
            }

            NaiveBayes m = new NaiveBayes();
            m.buildClassifier(cvtrain);
            Evaluation evaluation = new Evaluation(cvtrain);
            evaluation.evaluateModel(m, cvtest, new Object[0]);
            FastVector vec = evaluation.predictions();
            preds.add(vec);
            if (vals == null) {
                vals = DPLIB.getResults(vec);
            } else {
                double[] v2 = DPLIB.getResults(vec);
                for (int k = 0; k < v2.length; k++) {
                    vals[k] += v2[k];
                }
            }
        }
        vals = DPLIB.getMeasures(vals);
        System.out.print(name + ":" + file + ": ");
        System.out.println(Arrays.toString(vals));

        fout.write("\n" + name + ":" + file + ": " + "Vals=" + Arrays.toString(vals));

    }

    public static void Basic(Instances trainSeti, Instances testSeti, String file, FileWriter fout, String name) throws Exception {
        Instances trainSet = new Instances(trainSeti);
        Instances testSet = new Instances(testSeti);
        NaiveBayes l = new NaiveBayes();
        l.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(l, testSet, new Object[0]);
        FastVector vec = eval.predictions();
        double vals[] = DPLIB.getResults(vec);
        vals = DPLIB.getMeasures(vals);
        System.out.print(name + ":" + file + ": ");
        System.out.println(Arrays.toString(vals));

        fout.write("\n" + name + ":" + file + ": " + "Vals=" + Arrays.toString(vals));
    }

    public static void NNFilter(Instances trainSeti, Instances testSeti, String file, FileWriter fout, String name) throws Exception {

        Instances testSet = new Instances(testSeti);
        Instances trainSet = DPLIB.NNFilter(trainSeti, testSet, 10);

        NaiveBayes l = new NaiveBayes();
        l.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(l, testSet, new Object[0]);
        FastVector vec = eval.predictions();
        double vals[] = DPLIB.getResults(vec);
        vals = DPLIB.getMeasures(vals);
        System.out.print(name + ":" + file + ": ");
        System.out.println(Arrays.toString(vals));

        fout.write("\n" + name + ":" + file + ": " + "Vals=" + Arrays.toString(vals));

    }

    public static void WMulti(String[] files, String file, Instances testSeti, FileWriter fout, int features[], String name) throws Exception {

        ArrayList<String> train = new ArrayList<String>();
        for (String file2 : files) {
            if (file2.substring(0, 3).equals(file.substring(0, 3)) && file2.compareTo(file) < 0) {
                train.add(file2);
            }
        }

        if (train.size() > 0) {

            System.out.println("\n" + Arrays.toString(train.toArray()));
            Instances trainSet = DPLIB.LoadCSV(train, DPLIB.dp, features);
            Instances testSet = new Instances(testSeti);

            if (name.toLowerCase().contains("infogain")) {
                int indi[] = DPLIB.fSelectInfoGain(trainSet);
                if (DPLIB.useIterativeInfoGainSubsetting) {
                    indi = DPLIB.iterativeInfoGainSubsetting(trainSet, indi);
                } else {
                    indi = DPLIB.getTopX(indi);
                }
                trainSet = DPLIB.fSelectSet(trainSet, indi);
                testSet = DPLIB.fSelectSet(testSet, indi);
            }

            NaiveBayes l = new NaiveBayes();
            l.buildClassifier(trainSet);
            Evaluation eval = new Evaluation(trainSet);
            eval.evaluateModel(l, testSet, new Object[0]);
            FastVector vec = eval.predictions();
            double vals[] = DPLIB.getResults(vec);
            vals = DPLIB.getMeasures(vals);
            System.out.print(name + ":" + file + ": ");
            System.out.println(Arrays.toString(vals));

            fout.write("\n" + name + ":" + file + ": " + "Vals=" + Arrays.toString(vals));

        } else {
            System.out.print(name + ":" + file + ": ");
            System.out.println("!!! First Version. Copy from CV");
            fout.write("\n" + name + ":" + file + ": !!! First Version. Copy from CV");
        }
    }
}
