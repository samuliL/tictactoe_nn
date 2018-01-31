import java.util.HashMap;
/** Gradient class. 
* Encapsulates information needed to store the gradients. More convenient than dealing with two hashmaps of double arrays without encapsulation.
*/
public class Gradient{
  HashMap<Integer, double[][]> wGrad = new HashMap<>();
  HashMap<Integer, double[]> bGrad = new HashMap<>();
  
  /** Adds a given gradient to itself.
  * @param g The gradient to be added.
  * @param learningDirection Usually -1.0 or 1.0 denoting whether or not the gradient is added or substracted from this gradient. Can be used for scaling if necessary.
  */
  public void addToGradient(Gradient g, double learningDirection){
    for(Integer l : g.wGrad.keySet()){
      for(int i = 0; i < wGrad.get(l).length; i++){
        for(int j = 0; j < wGrad.get(l)[i].length; j++)
          wGrad.get(l)[i][j] += g.wGrad.get(l)[i][j]*learningDirection;
        bGrad.get(l)[i] += g.bGrad.get(l)[i]*learningDirection;
      }
    }
  }
}