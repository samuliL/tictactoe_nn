import java.util.Scanner;
import java.util.Objects;
import java.util.ArrayList;

/** Encapsulation of relevant methods and fields to run a game of tic-tac-toe.
  */

public class TicTacToe{
 
  /**The board represented as a two-dimensional array of integers.*/
  int[][] board;

  /**Variable that keeps track of how many turns have passed.*/
  int turnsPassed;
  /**The dimension of the game board.*/
  int dim; 
  /**Stores the amount of marks needed in a row to win the game.*/
  int inARow;
  
  /**The players of the game as instances of the class Player. @see Player*/
  Player player1, player2;
  
  /**Class' own method to check victory. 
  *@param lastMove A pair of integer coordinates for the most recent move.
  *@return Integer that represents the outcome of the game. -1 if game still ongoing, 0 if draw or 1 if the last moving player has won.
  */
  public int checkVictory(int[] lastMove){
    return checkVictory(lastMove, inARow, board);
  }
  
  /**The static method to check victory. In particular used by the minmax AI to evaluate the state of the game. Checks all the possibilities how the last move could have caused a player to win.
  *@param lastMove A pair of integer coordinates for the most recent move.
  *@param inARow How many marks are needed in a row for winning.
  *@param board The state of the game board.
  *@return Integer that represents the outcome of the game. -1 if game still ongoing, 0 if draw or 1 if the last moving player has won.
  */  
  public static int checkVictory(int[] lastMove, int inARow, int[][] board){
    
    int x = lastMove[0], y = lastMove[1];
    int pl = board[x][y];
        
    // variable to store the segment length
    int segLen = 1; 
    
    // check horizontal line from last move
    x = lastMove[0]+1;
    while(x < board.length && board[x][y] == pl){x++; segLen++;}
    x = lastMove[0]-1;
    while(x >= 0 && board[x][y] == pl){x--; segLen++;}

    if(segLen >= inARow)
      return 1;
    
    segLen = 1;
    x = lastMove[0];
    y = lastMove[1]+1;

    // check vertical line from last move
    while(y < board.length && board[x][y] == pl){y++; segLen++;}
    y = lastMove[1]-1;
    while(y >= 0 && board[x][y] == pl){y--; segLen++;}
    
    if(segLen >= inARow)
      return 1;

    // check diagonal line y = x 
    segLen = 1;
    x = lastMove[0]+1; y = lastMove[1]+1;
    while(y < board.length &&  x < board.length && board[x][y] == pl){y++; x++; segLen++;}
    x = lastMove[0]-1; y = lastMove[1]-1;
    while(y >= 0 && x >= 0 && board[x][y] == pl) {y--; x--; segLen++;}
    
    if(segLen >= inARow)
      return 1;

    // check diagonal line y = -x 
    segLen = 1;
    x = lastMove[0]-1; y = lastMove[1]+1;
    while(y < board.length && x >= 0 && board[x][y] == pl){y++; x--; segLen++;}
    x = lastMove[0]+1; y = lastMove[1]-1;
    while(y >= 0 && x < board.length && board[x][y] == pl){y--; x++; segLen++;}
    
    if(segLen >= inARow)
      return 1;
      
    // check if the board is full
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++)
        if(board[i][j] == 0)
          return -1;
    
    return 0;
  }
  
  /** Constructor for the class TicTacToe. If one of the player types is neural network, the structure of the network is decided here.
  * @param dimension The dimension of the game board.
  * @param targetLength How many in a row a player needs to win.
  * @param p1 The type of player 1 as a string. @see Player
  * @param p2 The type of player 2 as a string. @see Player
  */
  public TicTacToe(int dimension, int targetLength, String p1, String p2){
    board = new int[dimension][dimension];
    turnsPassed = 0;
    dim = dimension;
    inARow = targetLength;

    player1 = new Player(p1);
    player2 = new Player(p2);
    
    
    // initialize the players if they need initializing
    
    if(Objects.equals(p1, "Neural network"))
      player1.initializeNN(new int[] {dimension*dimension, 20, dimension*dimension});

    if(Objects.equals(p2, "Neural network"))
      player2.initializeNN(new int[] {dimension*dimension, 20, dimension*dimension});

    if(Objects.equals(p1, "Minmax"))
      player1.initializeMM(targetLength, 1);

    if(Objects.equals(p2, "Minmax"))
       player2.initializeMM(targetLength, 2);
  }
  
  /** Method that runs a game of tic-tac-toe. 
  *@param show If true, displays the game on console turn by turn.
  *@return The number of winning player or 0 if draw.
  */
  public int play(boolean show){
     
    resetBoard();
    int vic = 0;
    Player temp = null;
    while(true){
      // variable pl = 0 if player 1 is moving, and 1 if player 2 is moving
      int pl = turnsPassed % 2;
      
      if(pl == 0)
        temp = player1;
      else 
        temp = player2;

      int[] m = null;
      
      // make sure the neural network AI gets the board in the right format.
      if(pl == 1 && player2.isNN()){
        m = temp.move(invertBoard());
      } else m = temp.move(board);
      
      board[m[0]][m[1]] = pl+1;
      turnsPassed++;

      if(show) printBoard();
      
      vic = checkVictory(m);
      
      if(vic == 1)
        return pl+1;
      if(vic == 0)
        return 0;
      }
  }
  
  /** Prints the state of the board to the console. */
  public void printBoard(){
    System.out.println("Turn " + turnsPassed);
    for(int y = 0; y < board.length; y++){
      String row = new String();
      for(int x = 0; x < board.length; x++){
        String c = new String();
        switch(board[x][y]){
          case 0: c = "."; break;
          case 1: c = "X"; break;
          case 2: c = "O"; 
        }
        row = row + c;
      }
      System.out.println(row);
    }
  }
  
  /**Inverts the board. Needed for giving the neural network the board such that it always sees itself as playing with X's.
  *@return A representation of the game board such that X's and O's are switched.
  */
  public int[][] invertBoard(){
    int[][] b = new int[board.length][board.length];
    
    for(int x = 0; x < board.length; x++)
      for(int y = 0; y < board.length; y++)
        if(board[x][y] != 0)
          b[x][y] = (board[x][y] == 1) ? 1 : 2;
    return b;
  }
  
  /**Resets the board to empty and sets passed turns to 0.*/
  public void resetBoard(){
    for(int i = 0; i < board.length; i++)
      for(int j = 0; j < board.length; j++)
        board[i][j] = 0;
    turnsPassed = 0;
  }
  
  
  /** Runs a game of tic-tac-toe and records the moves as GameRecord for training the neural network.
  *@return Statistics of the game. @see GameRecord
  */
  public GameRecord recordedPlay(){
    // variables to store game data
    // board states in a neural network input friendly format
    ArrayList<double[]> bStates = new ArrayList<double[]>();
    // moves made during the game
    ArrayList<int[]> moves = new ArrayList<int[]>();
    // the number of the player who made the corresponding move
    ArrayList<Integer> playingAs = new ArrayList<Integer>();
    
    // initialize the game
    resetBoard();
    int vic = 0;
    Player temp = null;
    
    while(true){
      // pl keeps track of which player is moving, 0 for player 1, 1 for player 2
      int pl = turnsPassed % 2;
      if(pl == 0)
        temp = player1;
      else 
        temp = player2;
      
      
      int m[] = null;
      
      // make sure the neural network player gets the board in the correct format, in other words we want the network to always see itself as playing with X's
      if(pl == 1 && player2.isNN()){
        m = temp.move(invertBoard());
      } else m = temp.move(board);
      
      // record the game states - for player 2 invert the board so that the neural network sees itself always as the player with X's
      if(pl == 0)
        bStates.add(NeuralNetwork.formatInput(board)); 
      if(pl == 1)
        bStates.add(NeuralNetwork.formatInput(invertBoard())); 
      
      // record the move
      moves.add(new int[]{m[0], m[1]});
      // record which player made the move
      playingAs.add(pl+1); 
      
      board[m[0]][m[1]] = pl+1;
      turnsPassed++;
      
      vic = checkVictory(m);
      if(vic == 1) 
        return new GameRecord(moves, bStates, playingAs, pl+1); 
      if(vic == 0)
        return new GameRecord(moves, bStates, playingAs, 0);
    }
  }   
}