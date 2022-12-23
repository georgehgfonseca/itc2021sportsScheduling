package br.ufop.itc2021.algorithm.neighborhood;

import br.ufop.itc2021.model.*;

import java.util.*;

/**
 * This abstract class represents a Move (or Neighborhood). The basic methods as well as several counters (for future
 * analysis) are included.
 *
 * @author Tulio Toffolo
 */
public abstract class Move {

    public final Instance inst;
    public final String name;
    public final Random random;

    protected Solution solution;
    protected boolean intermediateState = false;

    protected long deltaCost = 0;
    protected long initialTotalCost = Integer.MAX_VALUE;

    protected boolean inChain = false;
    protected int initialInfeas = Integer.MAX_VALUE;
    protected int initialFeasCost = Integer.MAX_VALUE;
    protected int weight = 1;

    // basic statistics for future analysis
    private long nIters = 0;
    private long nImprovements = 0;
    private long nSideways = 0;
    private long nWorsens = 0;
    private long nRejects = 0;


    /**
     * Instantiates a new Move.
     *
     * @param inst   the problem reference.
     * @param random the random number generator.
     */
    public Move(Instance inst, Random random) {
        this.inst = inst;
        this.random = random;
        this.name = "";
    }

    /**
     * Instantiates a new Move.
     *
     * @param inst     the problem reference.
     * @param random   the random number generator.
     * @param name     the name of this neighborhood (for debugging purposes).
     * @param weight the priority (priority) of this neighborhood structure. The larger the value, the higher the
     *                 priority.
     */
    public Move(Instance inst, Random random, String name, int weight) {
        this.inst = inst;
        this.random = random;
        this.name = name;
        this.weight = weight;
    }


    /**
     * This method must be called whenever the modification made by this move is accepted. It ensures that the solution
     * as well as other structures are updated accordingly.
     */
    public void accept() {
        assert intermediateState : "Error: calling accept() before calling doMove().";
        intermediateState = false;

        // updating counters
        if (deltaCost < 0) nImprovements++;
        else if (deltaCost == 0) nSideways++;
        else nWorsens++;
    }

    /**
     * This method returns does the move and returns the impact (delta cost) in the solution.
     *
     * @param solution the solution to be modified.
     * @return the impact (delta cost) of this move in the solution.
     */
    public long doMove(Solution solution) {
        assert hasMove(solution) : "Error: move " + name + " being executed with hasMove() = false.";
        assert !intermediateState : "Error: calling doMove before mandatory call to accept() or reject().";
        intermediateState = true;

        nIters++;
        this.solution = solution;
        initialInfeas = solution.getInfeas();
        initialFeasCost = solution.getFeasCost();
        initialTotalCost = solution.getCost();
        return deltaCost = Long.MAX_VALUE;
    }

    public long getDeltaCost() {
        return deltaCost;
    }

    /**
     * This method returns a boolean indicating whether this neighborhood can be applied to the current solution.
     *
     * @return true if this neighborhood can be applied to the current solution and false otherwise.
     */
    public boolean hasMove(Solution solution) {
        return true;
    }

    /**
     * This method must be called whenever the modification made by this move are rejected. It ensures that the solution
     * as well as other structures are updated accordingly.
     */
    public void reject() {
        assert intermediateState : "Error: calling reject() before calling doMove().";
        intermediateState = false;

        // updating counters
        nRejects++;
    }

    /**
     * This method is called whenever the neighborhood should be reset (mainly to avoid the need of creating another
     * object).
     */
    public void reset() { }


    // region simple getters and setters

    public int getWeight() {
        return weight;
    }

    public void setWeight(int weight) {
        this.weight = weight;
    }

    public long getNIters() {
        return nIters;
    }

    public long getNAccepts() {
        return nImprovements + nSideways + nWorsens;
    }

    public long getNImprovements() {
        return nImprovements;
    }

    public long getNSideways() {
        return nSideways;
    }

    public long getNWorsens() {
        return nWorsens;
    }

    public long getNRejects() {
        return nRejects;
    }

    public String toString() {
        return name;
    }

    // endregion getters and setters
}
