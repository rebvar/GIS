/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package istgis;

/**
 *
 * @author sehossei
 */
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import weka.core.Instances;

public class GA {

    static double mProb = 0.05;
    static int mCount = 1;
    
    
    public static double GetMeanFittness(ArrayList<IChrm> hmm,int count)
    {
        if (count == 0)
            count = hmm.size();
        double sum = 0;
        for (int i=0;i<count;i++)
        {
            sum+=hmm.get(i).getFitness();
        }
        return sum/count;
    }
    
    
    
    public static Instances Mutate(Instances ds) {
        
        
        double r2 = DPLIB.rnd.nextDouble();
        
        if (r2 <= mProb) 
        {
            
            Set<Integer> rands = new HashSet<Integer>();
            int i=0;
            
            
            
            while(i<mCount)
            {
                
                int r1 = DPLIB.rnd.nextInt(ds.numInstances());
                if (rands.size()==ds.numInstances())
                    return ds;
                
                if (rands.contains(r1))
                    continue;
                double instLabel = ds.instance(r1).classValue();
                
                rands.add(r1);
                ds.instance(r1).setClassValue(1 - instLabel);
                
                
                HashSet<Integer> set = DPLIB.FindAllSimilarInstancesIndexes(r1, ds);
                Iterator<Integer> iter = set.iterator();
                while (iter.hasNext())
                {
                    r1 = iter.next();
                    rands.add(r1);
                    ds.instance(r1).setClassValue(1 - instLabel);
                }
                i++;
            }
            
        }
        return ds;
    }
    
    public static int tornament(ArrayList<IChrm> hmm)
    {
        int[] vals = new int[2];
        for (int i=0;i<vals.length;i++)
            vals[i] = DPLIB.rnd.nextInt(hmm.size());
        int maxInd=-1;
        double maxFit = 0;
        for (int i=0;i<vals.length;i++)
        {
            if (hmm.get(i).getFitness()>maxFit)
            {
                maxFit = hmm.get(i).getFitness();
                maxInd = i;
            }
        }
        
        return vals[maxInd];
    }

    public static Instances[] crossOver(Instances ds1, Instances ds2) {
        
        int ss= ds1.numInstances();
        int point1 = DPLIB.rnd.nextInt(ss);
        int point2 = point1;
        
        ds1.randomize(DPLIB.rnd);
        ds2.randomize(DPLIB.rnd);
        Instances ds1c = new Instances(ds1, 0);
        Instances ds2c = new Instances(ds2, 0);
        
        
        
        
        for (int i = 0; i < point1; i++) {
            ds1c.add(ds1.instance(i));
            
        }

        for (int i = 0; i < point2; i++) {
            ds2c.add(ds2.instance(i));
            
        }
        
        for (int i = point1; i < ds1.numInstances(); i++) {
            
            ds2c.add(ds1.instance(i));
        }
        
        
        for (int i = point2; i < ds2.numInstances(); i++) {
            ds1c.add(ds2.instance(i));
            
        }
        
        
        HashSet<Integer> pSet = new HashSet<Integer>();
        
        for (int i=0;i< ds1c.numInstances();i++)
        {
            if (pSet.contains(i))
                continue;
            HashSet<Integer> tmp = DPLIB.FindAllSimilarInstancesIndexes(i, ds1c);
            double lbl = 0;
            Integer[] t = tmp.toArray(new Integer[0]);
            int index=-1;
            for(int j=0;j<t.length;j++)
            {
                
                index = t[j];
                lbl+=ds1c.instance(index).classValue();
                pSet.add(index);
            }
            
            
            lbl = lbl/(t.length);
            if (lbl>=0.5)
                lbl=1;
            else
                lbl=0;
            
            
            
            ds1c.instance(i).setClassValue(lbl);
            for(int j=0;j<t.length;j++)
            {
                
                index = t[j];
                ds1c.instance(index).setClassValue(lbl);
            }
        }
        
        pSet.clear();
        for (int i=0;i< ds2c.numInstances();i++)
        {
            if (pSet.contains(i))
                continue;
            HashSet<Integer> tmp = DPLIB.FindAllSimilarInstancesIndexes(i, ds2c);
            double lbl = 0;
            Integer[] t = tmp.toArray(new Integer[0]);
            int index;
            for(int j=0;j<t.length;j++)
            {
                
                index = t[j];
                lbl+=ds2c.instance(index).classValue();
                pSet.add(index);
            }
            
            lbl = lbl/(t.length);
            if (lbl>=0.5)
                lbl=1;
            else
                lbl=0;
            
            ds2c.instance(i).setClassValue(lbl);
            for(int j=0;j<t.length;j++)
            {
                
                index = t[j];
                ds2c.instance(index).setClassValue(lbl);
            }
        }
        
        return new Instances[]{ds1c, ds2c};
    }
    

}
