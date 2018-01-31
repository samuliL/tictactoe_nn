import java.util.ArrayList;

/** Gamerecord class encapsulates useful statistics about a game that can be used for training a neural network.
*/
public class GameRecord{
  
  /** Keeps track of the moves that are made during the game as pairs of integers. */
  ArrayList<int[]> move;
  /** Tells which player made the current move. Must be declared once a move is added. */
  ArrayList<Integer> playingAs;
  /** The board state at a given point of the game in a format that can be directly inputted to a network.*/
  ArrayList<double[]> board;
  
  /** Stores the outcome of the game. 0 - draw, 1 - player 1 wins, 2 - player 2 wins.*/
  int outcome;
  
  /** The constructor. Takes the corresponding data of a game and encapsulates it into a GameRecord object. 
   * @param m List of moves.
   * @param b List of board states.
   * @param pl List of which player made which move.
   * @param oc The outcome of the game.
   */
  GameRecord(ArrayList<int[]> m, ArrayList<double[]> b, ArrayList<Integer> pl, int oc){
    
    move = new ArrayList<int[]>();
    for(int i = 0; i < m.size(); i++)
      move.add(m.get(i));
    
    board = new ArrayList<double[]>();
    for(int i = 0; i < b.size(); i++)
      board.add(b.get(i));
  
    playingAs = new ArrayList<Integer>();
    for(int i = 0; i < pl.size(); i++)
      playingAs.add(pl.get(i));
    
        
    outcome = oc;
  }
}