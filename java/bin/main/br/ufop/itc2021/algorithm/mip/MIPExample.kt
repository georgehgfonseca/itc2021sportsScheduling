package br.ufop.itc2021.algorithm.mip

import mip.*
import java.util.*

class MIPExample(val random: Random) {

    init {
        val model = Model(solverName = "CBC") // use gurobi.. eu não consigo pq a licença venceu  :(
        val x = List(10) { model.addBinVar(name = "x($it)", obj = random.nextInt(1000)) }

        // making sum(x) >= 1
        model += x geq 1 named "eq1"

        // making sum(k * x) <= 5  (para k aleatório) - jeito 1
        model += x.sum { random.nextInt(5) * it } leq 5 named "eq2"

        // making sum(k * x) <= 5  (para k aleatório) - jeito 2
        model += x.sum { variable -> random.nextInt(5) * variable } leq 5 named "eq2"

        // making sum(k * x) <= 5  (para k aleatório) - jeito 3
        model.addConstr(x.sum { random.nextInt(5) * it } leq 5, "eq3")

        // making sum(k * x) <= 5  (para k aleatório) - jeito 4
        val lhs = LinExpr()
        for (it in x)
            lhs += random.nextInt(5) * x
        model.addConstr(lhs leq 5, "eq4")

        // solving linear relaxation
        model.optimize()

        // printing result - jeito 1
        for (it in x)
            print("${it.name} = ${it.x}")

        // printing result - jeito 2
        print(x.map { "${it.name} = ${it.x}\n" })
    }
}

fun main() {
    val x = MIPExample(Random())
}
