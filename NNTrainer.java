import java.util.ArrayList;

/** A class that encapsulates methods for training a neural network. Training consists of running and recording several games in a batch, after which the network is given each game position and the move that was made, and the outcome of the game. If the outcome of the move was that eventually the game was won, that move on that input is encouraged in the future. If the outcome of the game was a loss, the corresponding move on the input is discouraged. The "encouraging" and "discouraging" here refer to adding or substracting the gradient obtained by backpropagation on that input. This is essentially supervised learning.
*/

class NNTrainer{
  
  public static void main(String[] args){
    // A new network can be trained by calling the train() method with suitable parameters. 
        
    // The training scheme below seems to achieve the best training against a random player. The network structure is determined in the function train() and can be changed there.
    // The result of this training scheme is around 85% wins with X's and 60% wins with O's against a random opponent.
    
    // train the network against a random player in large batches and a large step size   
    train(new TicTacToe(3, 3, "Random", "Neural network"), null, 500, 200, .1, 1.0, -1.0, .5);
    train(new TicTacToe(3, 3, "Neural network", "Random"), "weights.txt", 500, 200, .1, 1.0, -1.0, .5);

    // same again, but smaller step and batch sizes  
    train(new TicTacToe(3, 3, "Random", "Neural network"), "weights.txt", 300, 1000, .01, 1.0, -1.0, .5);
    train(new TicTacToe(3, 3, "Neural network", "Random"), "weights.txt", 300, 1000, .01, 1.0, -1.0, .5);

    // again smaller step and batch sizes  
    train(new TicTacToe(3, 3, "Random", "Neural network"), "weights.txt", 100, 1000, .001, 1.0, -1.0, .5);
    train(new TicTacToe(3, 3, "Neural network", "Random"), "weights.txt", 100, 1000, .001, 1.0, -1.0, .5);
    
    // finally, display some statistics of the games by the neural network against a random player
    showGame(new TicTacToe(3, 3, "Neural network", "Random"), "weights.txt", null, 10000, false);
    showGame(new TicTacToe(3, 3, "Random", "Neural network"), null, "weights.txt", 10000, false);

  }

  /** Runs a given amount of games with desired parameters.
  *@param game The TicTacToe object containing the game parameters.
  *@param p1File The file to load player 1's neural network weights, null if the player is not a neural network or if new random network is wanted. 
  *@param p2File The file to load player 1's neural network nullweights, null if the player is not a neural network or if new random network is wanted.
  *@param num The number of games to run.
  *@param display True if the game should be displayed in console.
  */
  public static void showGame(TicTacToe game, String p1File, String p2File, int num, boolean display){
    if(p1File != null)
      game.player1.nn.loadFromFile(p1File);
    if(p2File != null)
      game.player2.nn.loadFromFile(p2File);

    int p1Wins = 0, p2Wins = 0;
    for(int k = 0; k < num; k++){
      int r = game.play(display);
      if(r == 1)
        p1Wins++;
      if(r == 2)
        p2Wins++;
    }
    System.out.println("Player 1 wins:" + p1Wins + ", draws: " + (num-p1Wins-p2Wins) + ", player 2 wins: " + p2Wins);    
  }

  /**
  * Train the network by having it play against an opponent that can be itself, random, human or AI. The game is played a given amount of times, after which the outcomes are used for computing a gradient. The gradient can be then used to optimize the network using a gradient descent algorithm (with mini batches).
  *@param game The TicTacToe instance that contains the game parameters, including the player types.
  *@param file The file used for the network weights. If null, a new random network is created and the resulting weights will be saved in file weights.txt.
  *@param batchSize The number of games played until the gradients are added up and used in gradient descent. 
  *@param numBatches How many iterations (batches) will be run.
  *@param learningRate The learning rate, in other words, the scaling of the gradient.
  *@param positiveMod The modifier that multiplies the gradient when the outcome of the game was positive. Preferably 1.0.
  *@param negativeMod The modifier that multiplies the gradient when the outcome of the game was negative. Preferably -1.0.
  *@param drawMod The modifier that multiplies the gradient when the outcome of the game was a draw. Preferably 0.5.
  */
  public static void train(TicTacToe game, String file, int batchSize, int numBatches, double learningRate, double positiveMod, double negativeMod, double drawMod){
    
    // train the neural network declared here, copy the result to the neural network player after each training step
    NeuralNetwork nn = new NeuralNetwork(new int[] {game.dim*game.dim, 20, 20, game.dim*game.dim});
    
    if(file != null && game.player1.isNN())
      game.player1.nn.loadFromFile(file);

    if(file != null && game.player2.isNN())
      game.player2.nn.loadFromFile(file);
      
    if(file != null)
      nn.loadFromFile(file);
    else
      file = "weights.txt";


    int bCounter = 0;
    
    int p1Wins = 0;
    int p2Wins = 0;
    
    while(bCounter < numBatches){

      ArrayList<GameRecord> gameRecords = new ArrayList<GameRecord>();
      
      while(gameRecords.size() < batchSize){
        GameRecord g = game.recordedPlay();
        if(g.outcome != 0 || drawMod != 0.0)
          gameRecords.add(g);
      }
      // train the network based on batch of games     
      
      double learningDirection;
      
      // initializes the gradient to have the correct number of variables
      Gradient grad = null;
      grad = nn.initializeGradient();      
      
      double bSize = 0.0;
      
      for(GameRecord g : gameRecords){
        learningDirection = drawMod;
        
        for(int i = 0; i < g.move.size(); i++){
          // check if the current game on record was won by player 1 or player 2, and 
          if((g.outcome == 1 && g.playingAs.get(i) == 1) || (g.outcome == 2 && g.playingAs.get(i) == 2))
            learningDirection = positiveMod;
          else if ((g.outcome == 1 && g.playingAs.get(i) == 2) || (g.outcome == 2 && g.playingAs.get(i) == 1))
            learningDirection = negativeMod;
          

          bSize += 1.0;
          // if the game outcome with input g.board.get(i) was positive, reinforce that part of the total gradient
          // if the outcome was negative, substract the corresponding gradient from the total gradient
          grad.addToGradient(nn.getGradient(g.board.get(i), g.move.get(i)[0]*game.board.length+g.move.get(i)[1]), learningDirection);  
        }
        if(g.outcome == 1)
          p1Wins++;
        else if(g.outcome == 2)
          p2Wins++;

      }         
      // take the learning step and save the result in a file
      nn.gradientStep(grad, 1.0*learningRate, bSize);
      nn.saveToFile(file);
      
      // if either of the players is a neural network, load the weights from the file
      if(game.player1.isNN()){
        game.player1.nn.loadFromFile(file);
      } 
      if(game.player2.isNN()){
        game.player2.nn.loadFromFile(file);
      } 
      
      bCounter += 1;  
      
      // output some statistics
      if(bCounter % 1 == 0)
        System.out.println(bCounter + " player 1 wins:" + p1Wins + ", draws: " + (batchSize-p1Wins-p2Wins) + ", player 2 wins: " + p2Wins);
      p1Wins = 0; p2Wins = 0;
      
      // every 100th iteration output a sample game
      
      //if(bCounter % 100 == 0)
      //  game.play(true);
    }    
  }  
}