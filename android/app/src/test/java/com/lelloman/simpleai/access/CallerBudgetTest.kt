package com.lelloman.simpleai.access

import org.junit.Assert.*
import org.junit.Test

class CallerBudgetTest {
    @Test fun limitsConcurrencyAndReleasesOnClose() {
        val budget = CallerBudget { 0L }
        val first = budget.acquire(1)!!
        assertNull(budget.acquire(1))
        val others = (2..4).map { budget.acquire(it)!! }
        assertNull(budget.acquire(5))
        first.close()
        assertNotNull(budget.acquire(5))
        others.forEach { it.close() }
    }
    @Test fun budgetResetsAfterWindowAndDoesNotEvictActiveCall() {
        var time = 0L
        val budget = CallerBudget { time }
        repeat(30) { budget.acquire(1)!!.close() }
        assertNull(budget.acquire(1))
        time = 60_000
        val lease = budget.acquire(1)!!
        time = 120_000
        assertNull(budget.acquire(1))
        lease.close()
        assertNotNull(budget.acquire(1))
    }
}
