/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package istgis;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author sehossei
 */
public class GIS {

    public int popSize = 30;
    public double chrmSize = 0.02D;

    public int numGens = 20;

    public int numParts = 5;

    public int sizeTopP = 2;

    public int count = 1;

    public FileWriter fout;

    public String file;

    public void GIS(Instances trainSeti, Instances testSeti, String name) throws Exception {
        long starttime, endtime;
        starttime = System.currentTimeMillis();

        ArrayList<FastVector> preds = new ArrayList();
        ArrayList<IChrm> pop = new ArrayList();
        

        Instances trainSet = new Instances(trainSeti);
        Instances testSet = new Instances(testSeti);
        pop.clear();

        int tstSize = testSet.numInstances();
        int partSize = (int) Math.floor(tstSize / numParts);
        preds.clear();
        double[] vals = null;
        ArrayList<ArrayList<Double>> diffs = new ArrayList<>();
        
        testSet.randomize(DPLIB.rnd);
        HashSet<Integer> uinds = new HashSet<Integer>();
        System.out.print ('#');
        for (int p = 0; p < numParts; p++) {
            ArrayList<Double> diffp = new ArrayList<>();
            System.out.print(String.valueOf(p) + " ");
            pop.clear();
            int start = p * partSize;
            int end = (p + 1) * partSize;
            if (end > tstSize) {
                end = tstSize;
            }
            if (p == numParts - 1) {
                end = tstSize;
            }

            Instances testPart = new Instances(testSet, 0);
            for (int t = start; t < end; t++) {
                testPart.add(testSet.instance(t));
            }

            Instances vSet = DPLIB.NNFilterMulti(trainSet, testPart, 3);

            int size = (int) (trainSet.numInstances() * chrmSize);

            for (int i = 0; i < popSize; i++) {
                Instances trSet = new Instances(trainSet, 0);
                int j = 0;
                while (j < size) {
                    int index = DPLIB.rnd.nextInt(trainSet.numInstances());
                    trSet.add(trainSet.instance(index));

                    if (!uinds.contains(index)) {
                        uinds.add(index);
                    }

                    j++;
                }
                NaiveBayes l = new NaiveBayes();
                l.buildClassifier(trSet);
                Evaluation evaluation = new Evaluation(trSet);
                evaluation.evaluateModel(l, vSet, new Object[0]);
                FastVector vec = evaluation.predictions();
                vals = DPLIB.getResults(vec);
                vals = DPLIB.getMeasures(vals);

                pop.add(new GIS_Chrm(trSet, vals));
            }
            pop = DPLIB.POP_SORT(pop);

            int cnt = 0;
            int g = 0;
            for (g = 0; g < numGens; g++) {
                ArrayList<IChrm> popNEW = new ArrayList();
                
                for (int i = 0; i < sizeTopP; i++) {
                    popNEW.add(pop.get(i));
                }
                for (int i = 0; i < pop.size() - sizeTopP; i += 2) {
                    int idx1 = 0;
                    int idx2 = 0;
                    while (idx1 == idx2) {
                        if (cnt >= 3) {
                            idx1 = DPLIB.rnd.nextInt(pop.size());
                            idx2 = DPLIB.rnd.nextInt(pop.size());
                        } else {
                            idx1 = GA.tornament(pop);
                            idx2 = GA.tornament(pop);
                            cnt++;
                        }
                    }
                    cnt = 0;
                    Instances ds1 = ((GIS_Chrm) pop.get(idx1)).ds;
                    Instances ds2 = ((GIS_Chrm) pop.get(idx2)).ds;

                    Instances[] ret = GA.crossOver(ds1, ds2);
                    ds1 = ret[0];
                    ds2 = ret[1];
                    ds1 = GA.Mutate(ds1);
                    ds2 = GA.Mutate(ds2);
                    popNEW.add(new GIS_Chrm(ds1, null));
                    popNEW.add(new GIS_Chrm(ds2, null));
                }
                for (int i = 0; i < popNEW.size(); i++) {
                    NaiveBayes l = new NaiveBayes();
                    l.buildClassifier(((GIS_Chrm) popNEW.get(i)).ds);
                    Evaluation evaluation = new Evaluation(((GIS_Chrm) popNEW.get(i)).ds);
                    evaluation.evaluateModel(l, vSet, new Object[0]);
                    FastVector vec = evaluation.predictions();
                    vals = DPLIB.getResults(vec);
                    vals = DPLIB.getMeasures(vals);

                    ((GIS_Chrm) popNEW.get(i)).fitness = vals;
                }
                popNEW = DPLIB.POP_SORT(popNEW);
                boolean exit = false;
                int countComp = 0;

                
                popNEW = DPLIB.CombinePops(pop, popNEW);
                double diff = Math.abs(GA.GetMeanFittness(pop, countComp) - GA.GetMeanFittness(popNEW, countComp));
                pop = popNEW;
                diffp.add(diff);
                if (diff < 1.0E-4D) {
                    exit = true;
                }
                if ((((GIS_Chrm) pop.get(0)).getFitness() > 0.0D) && (exit)) {
                    break;
                }
                exit = false;
            }
            diffs.add(diffp);
            ArrayList<Double> w = new ArrayList();
            if (count == 0) {
                count = pop.size();
            }
            for (int i = 0; i < count; i++) {
                NaiveBayes l = new NaiveBayes();
                Instances tds = ((GIS_Chrm) pop.get(i)).ds;
                Instances testPartI = testPart;
                l.buildClassifier(tds);
                Evaluation evaluation = new Evaluation(tds);
                evaluation.evaluateModel(l, testPartI, new Object[0]);
                FastVector vec = evaluation.predictions();
                if (preds.size() == count) {
                    ((FastVector) preds.get(i)).appendElements(vec);
                } else {
                    preds.add(vec);
                }
                w.add(Double.valueOf(((GIS_Chrm) pop.get(i)).getFitness()));
            }

        }

        vals = DPLIB.getResultsSet(preds, testSet, false);
        double vc[] = vals.clone();
        vals = DPLIB.getMeasures(vals);
        System.out.println();
        System.out.print(name + ":" + file + ": ");
        System.out.println(Arrays.toString(vals));

        String diffsStr = "";
        for (int i = 0;i<diffs.size();i++)
        {
            diffsStr+=Arrays.toString(diffs.get(i).toArray(new Double[1]))+"--";
        }
        endtime = System.currentTimeMillis();
        System.out.println("Time;;" + name + ";;" + file + ":" + String.valueOf(endtime - starttime));
        System.out.println("Indexes;;" + name + ";;" + file + ":" + String.valueOf(trainSet.numInstances()) + ";" + String.valueOf(uinds.size()) + ";" + String.valueOf(Math.round(10000 * uinds.size() / (float) trainSet.numInstances()) / 100.0));
        System.out.println("VC-Raw;;" + name + ";;" + file + ": " + Arrays.toString(vc));
        System.out.println("DIFFS;;" + name + ";;" + file + ": " + diffsStr);
        fout.write("\n" + name + ":" + file + ": " + "Vals=" + Arrays.toString(vals));
        fout.write("\nTime;;" + name + ";;" + file + ":" + String.valueOf(endtime - starttime));
        fout.write("\nIndexes;;" + name + ";;" + file + ":" + String.valueOf(trainSet.numInstances()) + ";" + String.valueOf(uinds.size()) + ";" + String.valueOf(Math.round(10000 * uinds.size() / (float) trainSet.numInstances()) / 100.0));
        fout.write("\nVC-Raw;;" + name + ";;" + file + ": " + Arrays.toString(vc));
        fout.write("\nDIFFS;;" + name + ";;" + file + ": " + diffsStr);

    }
}
