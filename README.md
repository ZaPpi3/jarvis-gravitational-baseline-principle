# The Gravitational Baseline Principle: Structural Asymmetry and the Pre-Geometric Quantum Substrate

This repository hosts the complete academic package for the preprint paper **"The Gravitational Baseline Principle"**. Included here are the compiled production PDF, raw REVTeX 4-2 LaTeX source files, numerical coordinate emergence engines, and print-quality publication figures.

The core framework identifies a fundamental structural asymmetry between General Relativity (GR) and Quantum Field Theory (QFT), modelling gravity not as a standard field interaction, but as the universal baseline background emerging natively from a quantum information substrate.

---

## 📄 Document and Manifest Assets

For quick access to the paper's contents, structural layouts, or code baselines, navigate to the following core files:

*   **`main.pdf`**: The final, production-compiled document structured in the official, two-column Physical Review D (PRD) journal format.
*   **`Manuscript/`**: Contains the raw LaTeX source code and compilation configurations.
    *   `main.tex`: The unified, consolidated REVTeX 4-2 source file optimized for local compilation or direct upload to the arXiv production servers.
*   **`Figures/`**: Publication graphics dynamically embedded into the manuscript layout.
    *   `eigenmodes.png`: High-resolution (300 DPI) visual output showcasing spontaneous multi-dimensional coordinate emergence, embedded directly as Fig. 1.
*   **`Code/`**: The complete Python implementation of the numerical models supporting the theoretical framework.
    *   `eigenmodes.py`: Python script that builds the relational quantum entanglement kernel and diagonalises the low-lying Laplacian spectrum.

---

## 🌌 Core Theoretical Architecture

1. **The Structural Asymmetry**: QFT fundamentally requires a fixed kinematic background to define field operator algebras. GR denies a prior container, rendering the metric dynamical. This architectural mismatch blocks conventional perturbative quantization.
2. **The Baseline Principle**: Gravity is the universal framework that makes field-theoretic relations definable. It cannot be quantized as a peer gauge field without destroying background independence.
3. **Spectral Manifold Emergence**: Space and metric tensors co-emerge from the entanglement spectrum of a relational density matrix. Modifying the underlying quantum information mapping is macroscopically equivalent to modifying the metric tensor ($g_{\mu\nu}$).

---

## 🚀 Execution and Local Compilation

### Compiling the TeX Source Locally
The raw manuscript is written using the official American Physical Society **REVTeX 4-2** document class. To compile the source into the final production PDF locally, ensure the image files are kept in the same directory as the source and run:

```bash
cd Manuscript
pdflatex main.tex
```

### Running the Infrastructure Code
Generate your print-quality manifold graphics and reproduce the paper's figures using Python 3:

```bash
python Code/eigenmodes.py
```

---

## ✒️ Citation / BibTeX

```bibtex
@article{jarvis2026baseline,
  title={The Gravitational Baseline Principle: Structural Asymmetry and the Pre-Geometric Quantum Substrate},
  author={Jarvis, Paul},
  url={https://github.com/ZaPpi3/jarvis-gravitational-baseline-principle}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete documentation. All code scripts and paper text assets are open-source and free to adapt with appropriate attribution.
