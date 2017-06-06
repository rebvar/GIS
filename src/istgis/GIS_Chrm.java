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

import weka.core.Instances;

public class GIS_Chrm  implements IChrm{
    public Instances ds;
    public double[] fitness;
    
    public GIS_Chrm(Instances ds, double[] fitness)
    {
        this.ds=ds;
        this.fitness = fitness;
    
    }    
    
    @Override
    public double getFitness()
    {
        return getCustom();
    }

    @Override
    public double getCustom() {
        
        return getF()*getGMEAN();
        
    }
    
    @Override
    public double getGMEAN() {
        return Math.sqrt(fitness[1]*fitness[0]);
    }
    
    
    @Override
    public double getF() {
        return fitness[3];
    }
}