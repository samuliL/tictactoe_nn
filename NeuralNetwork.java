import java.util.concurrent.ThreadLocalRandom;
import java.util.Objects;
import java.lang.Math;
import java.util.ArrayList;
import java.util.HashMap;
import java.io.*;

/** Representation of a neural network with any number of layers.
*
*    The class is designed so that the hidden layers are assumed to have either sigmoid or rectified linear units (ReLUs) as activations and the output layer activation is the softmax function.
*    Contains an implementation of gradient descent algorithm using backpropagation for optimizing the weights.
*/
   
public class NeuralNetwork{

  /**  
  *  Layer class. Contains weights and biases of a layer.
  */
  private class Layer{
     
    double[][] weights;
    double[] biases;
    
    /** Represents the type of a neuron as a string, in this application we have three possibilities, "Softmax" (to get a probability distribution as output), "Sigmoid" and "ReLU" (rectified linear unit). */
    String type;
    
    /** Number of neurons in the layer. */
    int numNodes;
    /** Input dimension to the layer. Same as the number of neurons in the previous layer.*/
    int inputDim;
    
    /** Creates a new layer 
     * @param nodes The number of neurons in the layer.
     * @param dim The input dimension.
     * @param t The type as a string.
    */
    public Layer(int nodes, int dim, String t){
      numNodes = nodes;
      inputDim = dim;
      
      type = new String(t);
      weights = new double[nodes][dim];
      biases = new double[nodes];

      // initialize weights and biases with random gaussians
      for(int i = 0; i < nodes; i++){
        for(int j = 0; j < dim; j++)
          weights[i][j] = ThreadLocalRandom.current().nextGaussian();
        biases[i] = ThreadLocalRandom.current().nextGaussian();
      }  
    }
  }

  /** Layers of the network. */
  Layer[] layers; 
  
  /** Coefficient of the ReLU activation on the negative domain, i.e., ReLU(x) = (x > 0) ? x : RELU_NEG_COEFF*x; */
  final double RELU_NEG_COEFF = 0.3;
  
  /** A variable to store the input to the network after formatting it to be most compatible with the network.*/
  private double[] formattedInput = null;
  
  public void setFormattedInput(double[] d){
    formattedInput = new double[d.length];
    System.arraycopy( d, 0, formattedInput, 0, d.length );
  }
  
  public double[] getFormattedInput(){
    double[] a = new double[formattedInput.length];
    System.arraycopy( formattedInput, 0, a, 0, a.length );
    return a;
  }
  
  /** NeuralNetwork constructor. 
   * @param layerList is an array of integers that represent the sizes of layers. For example the list {25, 10, 5} means that there are 25 inputs to the first layer with 10 neurons, and the output layer has 5 neurons 
   */
  public NeuralNetwork(int[] layerList){
    layers = new Layer[layerList.length-1];
    
    // In this application we use the sigmoid function as the activation. The type can be changed here in order to experiment with ReLU activations.
    for(int i = 0; i < layerList.length-2; i++)
      layers[i] = new Layer(layerList[i+1], layerList[i], "Sigmoid");
    
    // Initialize the output layer to have a softmax activation.
    layers[layerList.length-2] = new Layer(layerList[layerList.length-1], layerList[layerList.length-2], "Softmax");
  }
 
  /** Formats the input to be suitable for the network.  
   * @param board The board as a two-dimensional int array where X's are represented by 1, O's by 2 and empty positions by 0.
   * @return A 1-dimensional double array where X's are represented by -1.0, O's by 1.0 and empty positions as 0.0's.
   */
  public static double[] formatInput(int[][] board){
    int normalization = 0;
    double[] out = new double[board.length*board.length];
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++){
        if(board[i][j] == 1){
          out[i*board.length + j] = -1.0;
        } else if(board[i][j] == 2){
          out[i*board.length + j] = 1.0;
        } else if(board[i][j] == 0) {
          out[i*board.length + j] = 0;
        } else {
          System.out.println("Board formatting mismatch. Board at [i][j]: " + board[i][j]);
          System.exit(0);
        }
        normalization += Math.abs(out[i*board.length + j]);
      }
    return out; 
  }  
  
  /** Computes the dot product of two input vectors.
  *  @param v1 First vector.
  *  @param v2 Second vector.
  *  @return A double value that is the dot product of the input vectors.
  */
  public static double dot(double[] v1, double[] v2){
    // returns the dot product of two given vectors
    if(v1.length != v2.length){
      System.out.println("Attempted to take the dot product of two vectors of different size.");
      System.exit(-1);
    }
    double rval = 0.0;
    for(int i = 0; i < v1.length; i++)
      rval += v1[i]*v2[i];
      
    return rval;
  }
  
  /** Computes the output of the network from given input.
  *  @param input The input as a double array. Must have the proper size.
  *  @return Output of the network.
  */
  public double[] feedForward(double[] input){
    for(int i = 0; i < layers.length; i++)
      input = activate(layers[i], input);
    
    return input;
  }
  
  /** Computes the activations of a given layer.
  * @param layer A layer of the network to be activated.
  * @param input The input to the layer.
  * @return A double array containing the activations of each neuron in the layer.
  */
  public double[] activate(Layer layer, double[] input){
    // returns the activations of a given layer with given input, new types of activations can be added to the end
    
    double[] output = new double[layer.numNodes];
    
    // Softmax activation
    if(Objects.equals(layer.type, "Softmax")){
      double s = 0.0;
      for(int i = 0; i < layer.numNodes; i++){
        output[i] = dot(layer.weights[i], input) + layer.biases[i];
        s += Math.exp(output[i]);
      }
      for(int i = 0; i < layer.numNodes; i++){
        double t = output[i];
        output[i] = Math.exp(output[i])/s;
      }
      return output;
      
      // ReLU activation
    } else if(Objects.equals(layer.type, "ReLU")) {
      for(int i = 0; i < layer.numNodes; i++){
        output[i] = dot(layer.weights[i], input) + layer.biases[i];
        output[i] = (output[i] > 0.0) ? output[i] : RELU_NEG_COEFF*output[i];
      }      
      return output;
      
      // Sigmoid activation
    } else if(Objects.equals(layer.type, "Sigmoid")) {
      for(int i = 0; i < layer.numNodes; i++){
        output[i] = dot(layer.weights[i], input) + layer.biases[i];
        output[i] = 1/(1+Math.exp(-output[i]));
      }      
      return output;       
    }
    System.out.println("Node type wrong in the network.");
    System.exit(-1);
    return null;
  }

  /** Initilizes a new Gradient object such that the resulting gradient's dimensions match the structure of the network.
  *  @return A new Gradient object of proper dimension, initialized to zero.
  */
  public Gradient initializeGradient(){
    Gradient g = new Gradient();
    for(int l = 0; l < layers.length; l++){
      g.wGrad.put(l, new double[layers[l].numNodes][layers[l].inputDim]);
      g.bGrad.put(l, new double[layers[l].numNodes]);
    }
    return g;    
  }  
  
  /** Takes a gradient step to a given direction.
  * @param g The gradient. Substracted from the weights and biases of the network.
  * @param lRate The learning rate.
  * @param batchSize The size of the batch if the gradient is a sum of several vectors (mini batch) and has not been normalized. 
  */
  public void gradientStep(Gradient g, double lRate, double batchSize){
    
    double s = 0.0;
    
    for(int l = 0; l < layers.length; l++)
      for(int i = 0; i < layers[l].numNodes; i++){
        for(int j = 0; j < layers[l].inputDim; j++){
          layers[l].weights[i][j] -= lRate*g.wGrad.get(l)[i][j]/batchSize;
        }  
        layers[l].biases[i] -= lRate*g.bGrad.get(l)[i]/batchSize;
      }
  }

  /** Computes the gradient of the network when the loss function is the negative log propability of the output given by outputNumber.
  * @param input The input to the network.
  * @param outputNumber The objective function is the negative log probability of the output given by this parameter.
  * @return Gradient of the loss function with respect to the weights and biases of the network.
  */
  public Gradient getGradient(double[] input, int outputNumber){
    Gradient g = new Gradient();

    // store the activations of the neurons for each layer
    ArrayList<double[]> activations = new ArrayList<double[]>();
    
    // temporary variables for the gradient values
    HashMap<Integer, double[][]> weightGrad = new HashMap<Integer, double[][]>();
    HashMap<Integer, double[]> biasGrad = new HashMap<Integer, double[]>();
    
    // copy the input to a new variable and feed forward to compute the activations
    double[] netWorkOutput = new double[input.length];
    for(int i = 0; i < input.length; i++) netWorkOutput[i] = input[i];
    for(int i = 0; i < layers.length; i++) {netWorkOutput = activate(layers[i], netWorkOutput); activations.add(netWorkOutput);}
    
    // variables to store the backpropagation coefficients
    double[] oldDelta = null, delta = null;
    
    // iterate over the layers starting from the output layer and moving towards the input layer
    for(int l = layers.length-1; l >= 0; l--){
      double[][] wGrad = new double[layers[l].numNodes][layers[l].inputDim];        
      double[] bGrad = new double[layers[l].numNodes];
      delta = new double[layers[l].numNodes];
        
      for(int i = 0; i < layers[l].numNodes; i++){
        // compute delta, the first delta comes from the fact that the network loss function is log(a) where a is the activation of the last layer's neuron outputNumber
        if(l == layers.length-1)
          delta[outputNumber] = -1.0/(activations.get(l)[outputNumber]);        
        else {
          for(int r = 0; r < layers[l+1].numNodes; r++)
            delta[i] += oldDelta[r]*getActivationsDerivative(l, r, i, activations);
        }
        
        // compute the gradients for weights
        for(int j = 0; j < layers[l].inputDim; j++){
          if(Objects.equals(layers[l].type, "Softmax")){
            // handle the output layer where the activation is the softmax function (the output layer only)
            wGrad[i][j] = (i == outputNumber) ? (activations.get(l)[i]-1)*activations.get(l-1)[j] : activations.get(l)[i]*activations.get(l-1)[j];
          } else if(Objects.equals(layers[l].type, "ReLu")) {
            // hidden layers with ReLU activations
            if(l > 0)
              wGrad[i][j] = delta[i]*((activations.get(l)[i] > 0) ? activations.get(l-1)[j] : RELU_NEG_COEFF*activations.get(l-1)[j]);
            else
              wGrad[i][j] = delta[i]*((activations.get(l)[i] > 0) ? input[j] : input[j]*RELU_NEG_COEFF);
          } else if(Objects.equals(layers[l].type, "Sigmoid")) {
            // hidden layers with sigmoid activations
            if(l > 0)
              wGrad[i][j] = delta[i]*activations.get(l)[i]*(1-activations.get(l)[i])*activations.get(l-1)[j];
            else
              wGrad[i][j] = delta[i]*activations.get(l)[i]*(1-activations.get(l)[i])*input[j];
          }
        }
        // compute the bias gradient for bias[i] when the activation is softmax
        if(Objects.equals(layers[l].type, "Softmax"))
          bGrad[i] = (i == outputNumber) ? 1-activations.get(l)[i] : activations.get(l)[i];
        else if(Objects.equals(layers[l].type, "ReLu")) 
          bGrad[i] = delta[i]*((activations.get(l)[i] > 0) ? 1.0 : RELU_NEG_COEFF);     
        else if(Objects.equals(layers[l].type, "Sigmoid"))
          bGrad[i] = delta[i]*activations.get(l)[i]*(1-activations.get(l)[i]);     
      }
      // update the deltas
      oldDelta = new double[delta.length];
      System.arraycopy( oldDelta, 0, delta, 0, delta.length );

      // add computed gradients to the list
      g.wGrad.put(l, wGrad);
      g.bGrad.put(l, bGrad);
    }
    return g;    
  }
  
  /** Computes the derivative of the activations of the layer l+1 with respect to the activations of the layer l.
  * @param l The number of the layer that is treated as a variable. Hence, the derivative of the activation on layer l+1 is computed.
  * @param r The number of the neuron at layer l+1. The activation of this neuron is treated as the function.
  * @param i the number of the neuron at layer l. The activation of this neuron is treated as the variable.
  * @param activations. The pre-computed numerical values of the activations in the network.
  * @return The derivative as a numerical double value.
  */
  private double getActivationsDerivative(int l, int r, int i, ArrayList<double[]> activations){
    double output = 0.0;
    
    if(Objects.equals(layers[l+1].type, "Softmax")){
      for(int q = 0; q < layers[l+1].numNodes; q++){
        output += layers[l+1].weights[q][i]*((r == q) ? activations.get(l+1)[r]*(1-activations.get(l+1)[r]): -activations.get(l+1)[r]*activations.get(l+1)[q]);
      }
    } 
    else if(Objects.equals(layers[l+1].type, "ReLu")){
      output = (activations.get(l+1)[r] >= 0) ? layers[l+1].weights[r][i] : layers[l+1].weights[r][i]*RELU_NEG_COEFF;
    } 
    else if(Objects.equals(layers[l+1].type, "Sigmoid")){
      output = activations.get(l+1)[r]*(1-activations.get(l+1)[r])*layers[l+1].weights[r][i];
    }
    
    return output;
  }
  
  /** Saves the weights and biases of the network in a file.
  * @param filename The name of the file to save into.
  */
  public void saveToFile(String filename){
    try{
      File out = new File(filename);
      FileWriter fw = new FileWriter(out);
      fw.write(Integer.toString(layers.length) + "\n");
      for(int l = 0; l < layers.length; l++){
        fw.write(Integer.toString(layers[l].weights.length) + "\n");
        fw.write(Integer.toString(layers[l].weights[0].length) + "\n");
        fw.write(layers[l].type + "\n");
        for(int i = 0; i < layers[l].weights.length; i++){
          for(int j = 0; j < layers[l].weights[i].length; j++)
            fw.write(Double.toString(layers[l].weights[i][j]) + "\n");
          fw.write(Double.toString(layers[l].biases[i]) + "\n");
        }
      }
      fw.close();
    } catch(IOException e) {
      System.err.format("Exception occurred trying to write to '%s'.", filename);
      e.printStackTrace();
    } 
  }

  /** Loads the weights and biases of the network from a file.
  * @param filename The name of the file to load from.
  */
  public void loadFromFile(String filename){
    try{
      FileReader in = new FileReader(filename);
      BufferedReader reader = new BufferedReader(in);
      
      layers = new Layer[Integer.parseInt(reader.readLine())];
      for(int l = 0; l < layers.length; l++){
        int n = Integer.parseInt(reader.readLine());
        int d = Integer.parseInt(reader.readLine());
        layers[l] = new Layer(n, d, reader.readLine());
        for(int i = 0; i < layers[l].weights.length; i++){
          for(int j = 0; j < layers[l].weights[i].length; j++)
            layers[l].weights[i][j] = Double.parseDouble(reader.readLine());
          layers[l].biases[i] = Double.parseDouble(reader.readLine());
        }
      }
    }
    catch (Exception e)
    {
      System.err.format("Exception occurred trying to read '%s'.", filename);
      e.printStackTrace();
    } 
  }
}
  