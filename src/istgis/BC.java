/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package istgis;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 *
 * @author sehossei
 */
public class BC {

    public static void RunTests()
            throws Exception {

        DPLIB.useIterativeInfoGainSubsetting = true;
        int iters = 30;

        int folds = 10;

        GIS gis = new GIS();

        NaiveBayes temp = new NaiveBayes();
        FileWriter fout = new FileWriter(new File("14-FOUTGENRES---" + "iters=" + String.valueOf(iters) + "-POPSIZE=" + String.valueOf(gis.popSize) + "-NumParts=" + String.valueOf(gis.numParts) + "-NumGens=" + String.valueOf(gis.numGens) + "-CHRMSize=" + String.valueOf(gis.chrmSize) + "-sizeTop=" + String.valueOf(gis.sizeTopP) + "-Learner=" + temp.getClass().getSimpleName() + ".txt"));
        gis.fout = fout;

        String startTST = "";
        String[] files = DPLIB.getFiles(DPLIB.dp);
        for (String file : files) {
            if (file.compareTo(startTST) >= 0) {
                ArrayList<String> train = new ArrayList();
                for (String file2 : files) {
                    if (!file2.substring(0, 3).equals(file.substring(0, 3))) {
                        train.add(file2);
                    }
                }

                gis.file = file;

                ArrayList<String> test = new ArrayList();
                test.add(file);
                Instances trainSetAll = DPLIB.LoadCSV(train, DPLIB.dp, DPLIB.featuresALL);
                Instances testSetAll = DPLIB.LoadCSV((ArrayList) test, DPLIB.dp, DPLIB.featuresALL);

                Instances trainSetckloc = DPLIB.LoadCSV(train, DPLIB.dp, DPLIB.featuresCKLOC);
                Instances testSetckloc = DPLIB.LoadCSV((ArrayList) test, DPLIB.dp, DPLIB.featuresCKLOC);

                int indi[] = DPLIB.fSelectInfoGain(trainSetAll);
                int indis2[] = null;
                if (DPLIB.useIterativeInfoGainSubsetting) {
                    indis2 = DPLIB.iterativeInfoGainSubsetting(trainSetAll, indi);
                } else {
                    indis2 = DPLIB.getTopX(indi);
                }

                Instances trainSetInfoGain = DPLIB.fSelectSet(trainSetAll, indis2);
                Instances testSetInfoGain = DPLIB.fSelectSet(testSetAll, indis2);

                gis.file = file;
                gis.fout = fout;

                int c = 0;

                //System.out.println(trainSetAll.numAttributes());
                //System.out.println(trainSetckloc.numAttributes());
                //System.out.println(trainSetInfoGain.numAttributes());
                //System.out.println("Selected Features:" + Arrays.toString(indis2));

                while (c < iters) {

                    //System.out.println(file + "-Iter:" + String.valueOf(c));
                    c++;
                    gis.GIS(trainSetckloc, testSetckloc, "GeneticCKLOC");
                    gis.GIS(trainSetAll, testSetAll, "GeneticA");
                    gis.GIS(trainSetInfoGain, testSetInfoGain, "GeneticInfoGain");

                    Benchmarks.WCFolds(testSetAll, folds, file, fout, "CVALIDATIONA");
                    Benchmarks.WCFolds(testSetckloc, folds, file, fout, "CVALIDATIONCKLOC");
                    Benchmarks.WCFolds(testSetAll, folds, file, fout, "CVALIDATIONInfoGain");

                    Benchmarks.Basic(trainSetAll, testSetAll, file, fout, "NAIVEA");
                    Benchmarks.Basic(trainSetckloc, testSetckloc, file, fout, "NAIVECKLOC");
                    Benchmarks.Basic(trainSetInfoGain, testSetInfoGain, file, fout, "NAIVEInfoGain");

                    Benchmarks.NNFilter(trainSetAll, testSetAll, file, fout, "NNFILTERA");
                    Benchmarks.NNFilter(trainSetckloc, testSetckloc, file, fout, "NNFILTERCKLOC");
                    Benchmarks.NNFilter(trainSetInfoGain, testSetInfoGain, file, fout, "NNFILTERInfoGain");

                    Benchmarks.WMulti(files, file, testSetAll, fout, DPLIB.featuresALL, "WMULTIA");
                    Benchmarks.WMulti(files, file, testSetckloc, fout, DPLIB.featuresCKLOC, "WMULTICKLOC");
                    Benchmarks.WMulti(files, file, testSetAll, fout, DPLIB.featuresALL, "WMULTIInfoGain");

                    System.out.println("----------------------------------------------");
                    fout.flush();
                }
                System.out.println("===================================================================");
            }
        }
        fout.close();
    }
}
