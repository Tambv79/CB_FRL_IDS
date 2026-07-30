# Baseline definitions

- Full-Contextual: all K candidates upload and are sample-weighted aggregated.
- Random-Budget: random ordering over the common static-eligible pool, hard residual-byte feasibility.
- Cost-Only: lowest update wire cost first.
- Utility-Only: positive marginal utility first.
- Utility-Cost-Ratio: positive marginal utility divided by update wire bytes.
- CB-Score: positive normalized utility minus 0.25 normalized cost, without a second cost division.
- Historical-DoubleCost: historical CB score divided by cost; retained only to test double penalization.
- Oort-Style-Adapted: positive marginal utility multiplied by an Oort-inspired system-duration penalty; the hard byte budget and current-metadata setting are adaptations, so this is not claimed as an exact reproduction of Oort.
- Oracle-Score-Exact: exact enumeration of all subsets for K=10, maximizing summed positive CB-Score under the exact byte budget.

The Oort adaptation follows the primary Oort principle of jointly considering statistical utility and system performance. It omits Oort's large-population exploration machinery because all ten candidates expose current metadata each round. This difference must be stated whenever results are reported.
