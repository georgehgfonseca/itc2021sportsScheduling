package br.ufop.itc2021.algorithm.heuristic;

import br.ufop.itc2021.algorithm.neighborhood.*;
import br.ufop.itc2021.model.*;
import br.ufop.itc2021.util.*;

import java.io.*;
import java.util.*;

/**
 * This class is a Step Counting Hill Climbing implementation.
 *
 * @author Tulio Toffolo
 */
public class SCHC extends Heuristic {

    /**
     * SCHC parameters.
     */
    private int stepSize;
    private long costBound;

    /**
     * Instantiates a new SCHC.
     *
     * @param inst  problem
     * @param random   random number generator
     * @param stepSize the SCHC step size value
     */
    public SCHC(Instance inst, Random random, int lowerBound, int stepSize ) {
        super(inst, random, "SCHC", lowerBound);

        // initializing the late acceptance list
        this.stepSize = stepSize;
        this.costBound = Integer.MAX_VALUE;
    }

    /**
     * Executes the SCHC.
     *
     * @param initialSolution the initial (input) solution.
     * @param timeLimitMillis the time limit (in milliseconds).
     * @param maxIters        the maximum number of iterations without improvements to execute.
     * @param output
     * @return the best solution encountered by the SCHC.
     */
    public Solution run(Solution initialSolution, long timeLimitMillis, long maxIters, PrintStream output) {
        long finalTimeMillis = System.currentTimeMillis() + timeLimitMillis;

        bestSolution = initialSolution;
        Solution solution = initialSolution.clone();

        costBound = bestSolution.getCost();

        int nItersWithoutImprovement = 0;
        int stepCounter = 0;

        while (System.currentTimeMillis() < finalTimeMillis && bestSolution.getCost() > lowerBound) {
            while (System.currentTimeMillis() < finalTimeMillis && nItersWithoutImprovement++ < maxIters && bestSolution.getCost() > lowerBound) {
                stepCounter++;

                Move move = selectMove(solution);
                double delta = move.doMove(solution);

                // if solution is improved...
                if (delta < 0) {
                    acceptMove(move);
                    nItersWithoutImprovement = 0;

                    if (solution.getCost() < bestSolution.getCost()) {
                        bestSolution = solution.clone();
                        Util.safePrintStatus(output, nIters, bestSolution, solution, "*");
                    }
                }

                // if solution is not improved, but is accepted...
                else if (delta == 0 || solution.getCost() <= costBound) {
                    acceptMove(move);
                }

                // if solution is rejected..
                else {
                    rejectMove(move);
                }

                // if necessary, updates costBound
                if (stepCounter >= stepSize) {
                    costBound = solution.getCost();
                    stepCounter = 0;
                }

                nIters++;
            }

            if (System.currentTimeMillis() < finalTimeMillis) {
                nItersWithoutImprovement = 0;
                costBound = initialSolution.getCost();
                stepCounter = 0;
                Util.safePrintText(output, "Restarting SCHC cost bound", "");
            }
        }

        return bestSolution;
    }


    /**
     * Returns the string representation of the heuristic.
     *
     * @return the string representation of the heuristic (with parameters).
     */
    public String toString() {
        return String.format("SCHC (stepSize=%s)", Util.longToString(stepSize));
    }
}
