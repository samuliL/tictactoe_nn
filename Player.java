import java.util.concurrent.ThreadLocalRandom;
import java.util.Objects;
import java.util.Scanner;
  /** Player class encapsulates the relevant methods and fields for the players of the game. Communicates with a human and neural network player, and contains the methods of a random and minmax AI.
  */

class Player{


  /** Variable that keeps track of how the player is controlled. */
  int type;
  
  /** If the player is controlled by a neural network, this field is used to store the network.*/
  NeuralNetwork nn = null;
  
  
  /** Variable for the minmax player for knowing how many to get in a row to win/lose.*/
  int minMaxInARow = 0;
  /** Variable for the minmax player for knowing its player number.*/
  int minMaxPlayer;
  
  /** Constructor. 
  * @param typeStr The type of the player as a string. Possibilities are "Human", "Neural Network, "Random" and "Minmax".
  */
  public Player(String typeStr){
    if(Objects.equals(typeStr, "Human")){
      type = 0;
    } else if (Objects.equals(typeStr, "Neural network")){
      type = 1;
    } else if (Objects.equals(typeStr, "Random")){
      type = 2;
    } else if (Objects.equals(typeStr, "Minmax")){
      type = 3;
    }
  }
  
  public boolean isNN(){return type == 1;}
  
  /** If the player is of type "Neural network", this method has to be called in order to initialize the network.
  * @param layers The layer structure of the network. @see NeuralNetwork
  */
  public void initializeNN(int[] layers){nn = new NeuralNetwork(layers);}
  
  /** If the player is of type "Minmax", this method has to be called in order to initialize the player.
  * @param inARow How many consecutive X's or O's are needed to win the game.
  * @param pl Which player the minmax is.
  */
  public void initializeMM(int inARow, int pl){
    minMaxInARow = inARow;
    minMaxPlayer = pl;
  }
  
  /** An umbrella method that the TicTacToe object can call to get a move regardless of the player type.
  *@param board The state of the board at the beginning of the turn for this player.
  *@return Pair of integers denoting the coordinates of the next move.
  */
  public int[] move(int[][] board){
    switch(type){
      case 0: return humanMove(board);
      case 1: return nnMove(board);
      case 2: return randomMove(board);
      case 3: return minmaxMove(board);
    }
    return null;
  }
  
  /** The move method for a random player. Picks an available square at random and returns that as the move.
  * @param board The state of the board before the move.
  * @return Pair of integers denoting the coordinates of the next move.
  */
  public int[] randomMove(int[][] board){
    while(true){
      int x = ThreadLocalRandom.current().nextInt(0, board.length);
      int y = ThreadLocalRandom.current().nextInt(0, board.length);
      if(board[x][y] == 0)
        return new int[] {x, y};
    }
  }

  /** The move method for a human player. Asks for input on the console as two integers and checks the validity of the inputted move. 
  * @param board The state of the board before the move.
  * @return Pair of integers denoting the coordinates of the next move.
  */
  public int[] humanMove(int[][] board){
    Scanner scan = new Scanner(System.in);
    int x = 0, y = 0;
    while(true){
      System.out.println("Input move as pair of integers in the range 0-"+(board.length-1)+":");
      x = Integer.parseInt(scan.nextLine());
      y = Integer.parseInt(scan.nextLine());
      if(x >= 0 && x < board.length && y >= 0 && y < board.length && board[x][y] == 0){
        return new int[] {x, y};
      }else{
        System.out.println("Illegal move.");
      }
    }
  }  
  
  /** The move method for a neural network player. Inputs the state of the board to the network and gets a distribution of moves that the network proposes. The distribution is conditioned on legal moves and a move is chosen according to the distribution.
  * @param board The state of the board before the move.
  * @return Pair of integers denoting the coordinates of the next move.
  */
  public int[] nnMove(int[][] board){
    if(nn == null){
      System.out.println("Neural network not initialized. Exiting.");
      System.exit(-1);
    } 
    
    // format the input to a neural network friendly format
    double[] input = NeuralNetwork.formatInput(board);
    
      
    double[] output = nn.feedForward(input);
      
    // condition the distribution "output" to legal moves by setting the probability of illegal moves to 0 and normalizing
    double d = 0.0;
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++){
        if(board[i][j] != 0)
          output[i*board.length + j] = 0.0;
        d += output[i*board.length + j];
      }
      
    for(int i = 0; i < output.length; i++)
      output[i] /= d;
    
    // sample from the conditioned distribution
    int[] move = nnSample(output, board.length);
    
    return move;
  }
  
  /** Samples from a discrete distribution representing the board.
  * @param distr A distribution where each element corresponds to a probability of making a move to the corresponding square on the board.
  * @param bSize The board size in order to map the vector distr to the two-dimensional array board[][].
  * @return A pair of integers representing the coordinates of the sampled move.
  */
  
  public int[] nnSample(double[] distr, int bSize){
    double[] cumul = new double[distr.length];
    cumul[0] = distr[0];
    for(int i = 1; i < distr.length; i++)
      cumul[i] = cumul[i-1] + distr[i];
    
    double rand = ThreadLocalRandom.current().nextDouble(0.0, 1.0);
    int idx = 0;
    
    // find the index where random number hits (this could be optimized for speed with binary search)
    while(cumul[idx] <= rand)
      idx++;
    
    int x = idx / bSize;
    int y = idx % bSize;
    return new int[]{x, y};
  }  
  
  /** The move method for a minmax player. Calculates recursively what is the best move to make by suggesting a move, then seeing how the opponent would optimally play with that board. The implementation is quite slow and could be easily sped up by saving the game states in a lookup table.
  * @param board The state of the board before the move.
  * @return Pair of integers denoting the coordinates of the next move.
  */
  public int[] minmaxMove(int[][] board){
    
    // copy the board to a new variable so we do not mess the actual board
    // also check if the board is empty - if yes, make a random move and save time (this is not optimal, but better for training the network)
    int s = 0;
    int[][] newBoard = new int[board.length][board.length];
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++){
        newBoard[i][j] = board[i][j];
        s += board[i][j];
      }
    if(s == 0)
      return randomMove(board);
      
    // stores the value of the best found move
    int maxVal = -2;
    // stores the best found move
    int[] bestMove = new int[]{-1,-1};
    
    for(int i = 0; i < board.length; i++){
      for(int j = 0; j < board.length; j++){
        if(board[i][j] == 0){
          // see what happens if move here
          newBoard[i][j] = minMaxPlayer;
          if(TicTacToe.checkVictory(new int[]{i, j}, minMaxInARow, newBoard) >= 0)
            return new int[]{i, j};
          
          int m = -minMaxMoveAux(newBoard, minMaxPlayer % 2 + 1);
          if(m > maxVal){
            bestMove = new int[]{i, j};
            maxVal = m;
          }
          newBoard[i][j] = 0;
        }
      }
    }  
    return bestMove;
  }
  /**
  * An auxiliary function for computing the best move recursively. Does not keep track of what moves are made, and only returns the optimal value. In other words, whether or not optimal play will result in a draw, victory or loss for the given player.
  *@param board The state of the board.
  *@param pl The player whose turn it is.
  *@return The "value" of the current state of the board. Returns 0 if the optimal play results in a draw, 1 if win and -1 if loss.
  */
  public int minMaxMoveAux(int[][] board, int pl){
    // checks if player pl wins, if not recursively tries next moves
    int maxVal = -1;
    int res = 0;
    
    int[][] newBoard = new int[board.length][board.length];
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++)
        newBoard[i][j] = board[i][j];
        
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++){
        if(board[i][j] == 0){
          // see what happens if move here
          newBoard[i][j] = pl;

          res = TicTacToe.checkVictory(new int[]{i, j}, minMaxInARow, newBoard);
          if(res == 0)
            return 0;
          if(res == 1)
            return 1;
          
          int m = -minMaxMoveAux(newBoard, pl % 2 + 1);
          
          if(m > maxVal)
            maxVal = m;
          newBoard[i][j] = 0;
        }
      }
    return maxVal;
  }
}