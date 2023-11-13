# Simple CLI-based portfolio re-balancing tool

When investing in multiple assets, it is common to specify target percentages for each asset.
As prices fluctuate, these percentages are violated over time.
To keep a steady risk profile, when investing new money, this payment should be distributed among the assets such
that the new percentages are as close as possible to the original ones.

Basically the convex optimization problem
$$\min ||desired - (current + payment \cdot x)||_2 \quad  s.t. x \geq 0, \; sum(x)=1$$
has to be solved. This is done using [cvxpy](https://www.cvxpy.org/), a convex optimization library.