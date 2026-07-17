# The Gravitational Baseline Principle

**Structural Asymmetry and the Pre-Geometric Quantum Substrate**

Paul Jarvis · Independent Researcher, United Kingdom
[mrpaulwjarvis@gmail.com](mailto:mrpaulwjarvis@gmail.com) · ORCID: [0009-0009-8933-857X](https://orcid.org/0009-0009-8933-857X)

📄 [Read the paper (PDF)](main.pdf) · 📝 [LaTeX source](main.tex) · 💻 [Reproducibility code](Code/)

---

## Summary

General Relativity and Quantum Field Theory sit on incompatible foundations:
QFT requires a fixed background to define locality, while GR makes the metric
itself dynamical. This paper formalizes that tension as the **Gravitational
Baseline Principle**: gravity is not a peer interaction among fields but the
universal geometric framework that renders all field-theoretic relations
definable - not a field within a background, but the background itself.

The paper proposes a minimal, background-independent quantum substrate, built
from the mutual-information structure between subsystems of a relational
density matrix, from which smooth geometry, the graviton, and thermodynamic
gravitational dynamics are conjectured to co-emerge in the continuum limit.

## What the numerics actually show

Rather than treating the continuum and graviton limits as automatic, the
paper brings established theorems to bear on exactly what they require -
graph-Laplacian convergence theory (Belkin & Niyogi 2008; Singer 2006) for
the continuum limit, and the Fierz–Pauli uniqueness theorem for the graviton
- and then **tests both conditions directly**, by exact diagonalization on a
concrete, minimal realization of the substrate (the critical spin-1/2
antiferromagnetic Heisenberg chain, with a gapped/dimerized chain as a
control):

| Test | Result |
|---|---|
| Emergent Lorentz-consistent dispersion (Sec. 5) - is the low-energy excitation spectrum linear, not quadratic, in momentum? | **Confirmed.** Finite-size gap scales as $N^{-0.96}$ ($R^2=0.999994$), matching the exact des Cloizeaux–Pearson result; a quadratic (non-relativistic) alternative is decisively excluded ($R^2=0.009$). |
| Euclidean metric encoding (Sec. 4.1) - does the mutual-information matrix encode a genuine Euclidean distance, as the graph-Laplacian convergence theorems require? | **Falsified** for the critical substrate ($R^2=0.9998$ favors a logarithmic, not linear, fit) - but not featurelessly: the specific form obtained matches Calabrese–Cardy CFT entanglement scaling and points toward a hyperbolic, holographically-flavored alternative. The gapped control recovers the originally-hoped-for Euclidean behavior. |

In short: one open problem resolves in the framework's favor, the other is
redirected toward a specific, falsifiable, non-Euclidean alternative rather
than closed. See **[`Code/README.md`](Code/README.md)** for exact
reproduction instructions.

## Status and scope

This is a conceptual, analytical paper whose central claims are stated as
precise, falsifiable technical conditions and then partly tested numerically
(see above) rather than asserted outright. Section 9 (Limitations and
Outlook) lists the remaining open work: establishing the substrate's emergent
gauge redundancy, extending the numerical tests beyond the 1D Heisenberg
chain and toward higher-dimensional substrates, a Lorentzian extension,
derivation of matter field algebras, renormalization analysis, and
simulation of the substrate's own proposed (non-linear, self-consistent)
evolution equation, which the current tests do not simulate directly.

This is an independent research project, not peer-reviewed. Feedback and
replication attempts are welcome.

## Citing this work

If you reference this paper, please cite it as:

```
P. Jarvis, "The Gravitational Baseline Principle: Structural Asymmetry and
the Pre-Geometric Quantum Substrate" (2026).
```

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE)
file for details. In short: reuse, modification, and redistribution are
permitted, including commercially, provided the copyright notice is kept.
