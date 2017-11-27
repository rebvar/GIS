/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package istgis;

import java.io.File;
import weka.core.*;
import java.io.FileReader;
import java.util.*;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.neighboursearch.LinearNNSearch;

import weka.attributeSelection.*;
import weka.classifiers.bayes.NaiveBayes;

/**
 *
 * @author sehossei
 */
public class DPLIB {

    public static int[] featuresCKLOC;
    public static int[] featuresALL;
    public static String dp;
    static double cf = 0.5;

    public static double infoGainCount = 0.5;
    public static boolean useIterativeInfoGainSubsetting = true;

    public static Random rnd = new Random(System.currentTimeMillis());

    public static int find(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) {
                return i;
            }
        }
        return -1;

    }

    public static boolean findB(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) {
                return true;
            }
        }
        return false;
    }

    public static String[] getFiles(String path) {
        File dir = new File(path);
        ArrayList<String> ret = new ArrayList<String>();
        File[] filesList = dir.listFiles();
        for (File f : filesList) {
            if (f.isDirectory()) {
                continue;
            }
            if (f.isFile()) {
                if (f.getName().indexOf(".arff") > 0) {
                    ret.add(f.getName());
                } else {
                    continue;
                }
            }
        }
        return ret.toArray(new String[1]);
    }

    public static int[] getTopX(int indi[]) throws Exception {

        int count = (int) (indi.length * infoGainCount);
        if (indi[indi.length - 1] != indi.length - 1) {
            throw new Exception("Incorrect Attribute");
        }

        int indis2[] = new int[count + 1];
        for (int i = 0; i < count; i++) {
            indis2[i] = indi[i];
        }

        indis2[indis2.length - 1] = indi[indi.length - 1];

        return indis2;
    }

    public static Instances LoadCSV(String file, String dp, int ft[]) throws Exception {

        ArrayList<String> files = new ArrayList<String>();
        files.add(file);
        return DPLIB.LoadCSV(files, dp, ft);

    }

    public static Instances LoadCSV(ArrayList<String> files, String dp, int ft[]) throws Exception {
        int id = 0;
        try {
            Arrays.sort(ft);

            String f = files.get(0);
            FileReader file = new FileReader(dp + f);
            Instances insts = new Instances(file);
            int numAttrs = insts.numAttributes();
            insts.delete();
            int skipPercent = 0, loadPercent = 100;
            for (String f2 : files) {
                String f1 = f2.trim();
                file = new FileReader(dp + f1);
                Instances data = new Instances(file);

                int num = data.numInstances();
                for (int i = (int) (num * skipPercent / 100.0); i < (int) (num * loadPercent / 100.0); i++) {
                    //MyInstance tins = (MyInstance) data.instance(i);
                    //tins.id = id;                    
                    insts.add(data.instance(i));
                    id++;
                }

            }

            for (int i = 0; i < numAttrs - 1; i++) {
                int attrInd = numAttrs - 1 - 1 - i;

                if (!findB(ft, attrInd)) {
                    insts.deleteAttributeAt(attrInd);
                }
            }
            numAttrs = ft.length;

            int num = insts.numInstances();
            for (int i = 0; i < num; i++) {

                double[] lst = insts.instance(i).toDoubleArray();
            }

            num = insts.numInstances();
            for (int i = 0; i < num; i++) {
                if (insts.instance(i).value(numAttrs - 1) > 0) {
                    insts.instance(i).setValue(numAttrs - 1, 1);
                } else {
                    insts.instance(i).setValue(numAttrs - 1, 0);
                }
            }
            insts.setClassIndex(numAttrs - 1);
            return insts;

        } catch (Exception e) {
            System.out.println(e);
            System.out.println(files);
        }
        return null;
    }

    public static double[] getResults(weka.core.FastVector vec) {
        double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
        for (int p = 0; p < vec.size(); p++) {

            NominalPrediction np = (NominalPrediction) vec.elementAt(p);

            double geo = np.actual();
            double res = np.distribution()[1];

            if (geo >= cf) {
                geo = 1;
            } else {
                geo = 0;
            }

            if (res >= cf) {
                res = 1;
            } else {
                res = 0;
            }

            if (geo == 0 && res == 0) {
                tn += 1;
            } else if (geo > 0 && res > 0) {
                tp += 1;
            } else if (geo == 0 && res > 0) {
                fp += 1;
            } else if (geo > 0 && res == 0) {
                fn += 1;
            }

        }

        return new double[]{tp, tn, fp, fn};
    }
    
    public static double[] getResultsSet(ArrayList<FastVector> preds, Instances tss, boolean cluster) throws Exception {
        double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;

        int num = tss.numInstances();
        int numAttrs = tss.numAttributes();

        for (int i = 0; i < num; i++) {
            double geo = tss.instance(i).value(numAttrs - 1);

            ArrayList<Double> lss = new ArrayList<Double>();

            for (int k = 0; k < preds.size(); k++) {
                double v = ((NominalPrediction) preds.get(k).elementAt(i)).distribution()[1];

                lss.add(v);
            }

            double res = 0;
            if (cluster) {
                //Cluster the results. For ensembles. Not implemented
            } else {
                for (double val : lss) {
                    res += val;
                }
                res /= lss.size();

            }

            if (geo >= cf) {
                geo = 1;
            } else {
                geo = 0;
            }

            if (res >= cf) {
                res = 1;
            } else {
                res = 0;
            }

            if (geo == 0 && res == 0) {
                tn += 1;
            } else if (geo > 0 && res > 0) {
                tp += 1;
            } else if (geo == 0 && res > 0) {
                fp += 1;
            } else if (geo > 0 && res == 0) {
                fn += 1;
            }

        }

        return new double[]{tp, tn, fp, fn};
    }

    public static ArrayList<IChrm> POP_SORT(ArrayList<IChrm> hmm) {
        for (int i = 0; i < hmm.size() - 1; i++) {
            for (int j = i + 1; j < hmm.size(); j++) {
                if (hmm.get(i).getFitness() < hmm.get(j).getFitness()) {
                    IChrm tmp = hmm.get(i);
                    hmm.set(i, hmm.get(j));
                    hmm.set(j, tmp);
                }
            }
        }
        return hmm;
    }

    public static Instances fSelectSet(Instances set, int[] indices) throws Exception {
        Instances nS = new Instances(set);
        for (int i = nS.numAttributes() - 2; i >= 0; i--) {
            if (!findB(indices, i)) {
                nS.deleteAttributeAt(i);
            }
        }
        return nS;
    }

    public static int[] fSelectInfoGain(Instances train) throws Exception {

        AttributeSelection attsel = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();

        Ranker r = new Ranker();
        attsel.setSearch(r);
        attsel.setEvaluator(eval);

        attsel.SelectAttributes(train);

        int[] indices = attsel.selectedAttributes();

        return indices;
    }

    public static int[] iterativeInfoGainSubsetting(Instances trainSet, int ranks[]) throws Exception {
        int folds = 10;
        double f = 0;
        Instances trset = new Instances(trainSet);
        trset.stratify(folds);
        int count = 2;

        int tranks[] = null;
        while (count < ranks.length - 1) {
            double vals[] = null;

            for (int i = 0; i < folds; i++) {
                tranks = new int[count + 1];
                for (int j = 0; j < count; j++) {
                    tranks[j] = ranks[j];
                }

                tranks[count] = ranks[ranks.length - 1];
                Instances trCV = DPLIB.fSelectSet(trset.trainCV(folds, i), tranks);
                Instances tsCV = DPLIB.fSelectSet(trset.testCV(folds, i), tranks);
                NaiveBayes nb = new NaiveBayes();
                nb.buildClassifier(trCV);
                Evaluation eval = new Evaluation(trCV);
                eval.evaluateModel(nb, tsCV);
                FastVector vec = eval.predictions();

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
            if (f < vals[vals.length - 1]) {
                f = vals[vals.length - 1];
                count++;
            } else {
                break;
            }
        }
        return tranks;
    }

    public static ArrayList<IChrm> CombinePops(ArrayList<IChrm> hmm, ArrayList<IChrm> newHMM) {
        int i = 0;
        int j = 0;
        ArrayList<IChrm> ret = new ArrayList<IChrm>();
        while (ret.size() < hmm.size()) {
            if (hmm.get(i).getFitness() >= newHMM.get(j).getFitness()) {
                ret.add(hmm.get(i));
                i += 1;
            } else {
                ret.add(newHMM.get(j));
                j += 1;
            }
        }
        return ret;

    }

    public static double[] getMeasures(double[] vals) {
        double tp = vals[0], tn = vals[1], fp = vals[2], fn = vals[3];
        double prec = 0.0, accu = 0.0, recall = 0.0, gm = 0.0, bac = 0.0;
        if (tp + fn != 0) {
            recall = tp / (tp + fn);
        }

        if (tp + fp != 0) {
            prec = tp / (tp + fp);
        }

        accu = 0;

        double F = 0;
        if (prec + recall != 0) {
            F = (2.0 * prec * recall) / (prec + recall);
        }

        return new double[]{recall, prec, accu, F};

    }

    public static Instances List2Instances(ArrayList<Double> data) {

        int numInstances = data.size();
        int numDimensions = 1;
        FastVector atts = new FastVector();
        ArrayList<Instance> instances = new ArrayList<Instance>();
        for (int dim = 0; dim < numDimensions; dim++) {
            Attribute current = new Attribute("Attribute" + String.valueOf(dim), dim);
            if (dim == 0) {
                for (int obj = 0; obj < numInstances; obj++) {
                    instances.add(new SparseInstance(numDimensions));
                }
            }

            for (int obj = 0; obj < numInstances; obj++) {
                instances.get(obj).setValue(current, data.get(obj));
            }

            atts.addElement(current);
        }
        Instances newDataset = new Instances("Dataset", atts, atts.size());

        for (Instance inst : instances) {
            newDataset.add(inst);
        }
        return newDataset;
    }

    public static Instances NNFilter(Instances train, Instances test, int count) throws Exception {

        Instances trainCopy = new Instances(train);
        Instances testCopy = new Instances(test);
        trainCopy.setClassIndex(-1);
        testCopy.setClassIndex(-1);
        trainCopy.deleteAttributeAt(trainCopy.numAttributes() - 1);
        testCopy.deleteAttributeAt(testCopy.numAttributes() - 1);
        Instances out = new Instances(train, 0);
        LinearNNSearch knn = new LinearNNSearch(trainCopy);

        Set<Integer> instSet = new HashSet<Integer>();

        for (int i = 0; i < test.numInstances(); i++) {
            Instances nearestInstances = knn.kNearestNeighbours(testCopy.instance(i), count);

            for (int j = 0; j < nearestInstances.numInstances(); j++) {
                int ind = FindInstanceIndex(nearestInstances.instance(j), train);
                if (ind < 0) {
                    throw new Exception("Instance Not Found");
                } else {
                    if (instSet.contains(ind)) {
                        continue;
                    }
                    instSet.add(ind);
                    Instance finst = train.instance(ind);
                    out.add(finst);

                }
            }

        }
        return out;
    }

    public static Instances NNFilterMulti(Instances train, Instances test, int count) throws Exception {

        if (count == 0) {
            count = 1;
        }
        Instances trainCopy = new Instances(train);
        Instances testCopy = new Instances(test);
        trainCopy.setClassIndex(-1);
        testCopy.setClassIndex(-1);
        trainCopy.deleteAttributeAt(trainCopy.numAttributes() - 1);
        testCopy.deleteAttributeAt(testCopy.numAttributes() - 1);
        Instances out = new Instances(train, 0);
        LinearNNSearch knn = new LinearNNSearch(trainCopy);
        knn.setDistanceFunction(new EuclideanDistance(trainCopy));

        Set<Integer> instSet = new HashSet<Integer>();

        for (int i = 0; i < test.numInstances(); i++) {
            Instances nearestInstances = knn.kNearestNeighbours(testCopy.instance(i), count);
            for (int j = 0; j < nearestInstances.numInstances(); j++) {
                int ind = FindInstanceIndex(nearestInstances.instance(j), train);
                if (ind < 0) {
                    throw new Exception("Instance Not Found");
                } else {
                    if (instSet.contains(ind)) {
                        continue;
                    }
                    instSet.add(ind);
                    Instance finst = train.instance(ind);
                    out.add(finst);

                }
            }

        }

        knn.setDistanceFunction(new ChebyshevDistance(trainCopy));

        for (int i = 0; i < test.numInstances(); i++) {
            Instances nearestInstances = knn.kNearestNeighbours(testCopy.instance(i), count);
            for (int j = 0; j < nearestInstances.numInstances(); j++) {
                int ind = FindInstanceIndex(nearestInstances.instance(j), train);
                if (ind < 0) {
                    throw new Exception("Instance Not Found");
                } else {
                    if (instSet.contains(ind)) {
                        continue;
                    }
                    instSet.add(ind);
                    Instance finst = train.instance(ind);
                    out.add(finst);
                }
            }

        }

        knn.setDistanceFunction(new ManhattanDistance(trainCopy));

        for (int i = 0; i < test.numInstances(); i++) {
            Instances nearestInstances = knn.kNearestNeighbours(testCopy.instance(i), count);
            for (int j = 0; j < nearestInstances.numInstances(); j++) {
                int ind = FindInstanceIndex(nearestInstances.instance(j), train);
                if (ind < 0) {
                    throw new Exception("Instance Not Found");
                } else {
                    if (instSet.contains(ind)) {
                        continue;
                    }
                    instSet.add(ind);
                    Instance finst = train.instance(ind);
                    out.add(finst);
                }
            }

        }

        return out;
    }

    public static int FindInstanceIndex(Instance ins, Instances set) {
        for (int i = 0; i < set.numInstances(); i++) {
            boolean eq = true;
            for (int j = 0; j < ins.numAttributes(); j++) {
                if (ins.value(j) != set.instance(i).value(j)) {
                    eq = false;
                    break;

                }
            }
            if (eq) {
                return i;
            }
        }
        return -1;
    }

    public static HashSet<Integer> FindAllSimilarInstancesIndexes(int index, Instances set) {

        HashSet<Integer> indexSet = new HashSet<Integer>();
        Instance ins = set.instance(index);
        for (int i = 0; i < set.numInstances(); i++) {
            if (i == index) {
                indexSet.add(i);
                continue;
            }
            boolean eq = true;
            for (int j = 0; j < ins.numAttributes() - 1; j++) {
                if (ins.value(j) != set.instance(i).value(j)) {
                    eq = false;
                    break;

                }
            }
            if (eq) {
                indexSet.add(i);
            }
        }
        return indexSet;
    }
}
