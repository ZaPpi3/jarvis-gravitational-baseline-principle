# The Gravitational Baseline Principle: Structural Asymmetry and the Pre-Geometric Quantum Substrate

This repository hosts the complete academic package for the preprint paper **“The Gravitational Baseline Principle”**. Included here are the compiled production PDF, raw REVTeX 4‑2 LaTeX source files, numerical coordinate‑emergence engines, and print‑quality publication figures.

The core framework identifies a fundamental structural asymmetry between General Relativity (GR) and Quantum Field Theory (QFT), modelling gravity not as a standard field interaction, but as the universal baseline background emerging natively from a quantum information substrate.

---

## 📄 Document and Manifest Assets

For quick access to the paper's contents, structural layouts, or code baselines, navigate to the following core files:

### **main.pdf**  
The final, production‑compiled manuscript in the official two‑column Physical Review D (PRD) format.

### **Manuscript/**  
Contains the raw LaTeX source code and build configuration.

### **main.tex**  
The unified REVTeX 4‑2 source file, optimized for local compilation or direct arXiv submission.

### **Figures/**  
All publication graphics dynamically embedded into the manuscript.

- **eigenmodes.png** — High‑resolution (300 DPI) visualisation of spontaneous coordinate‑chart emergence from a Gaussian entanglement kernel (Fig. 1).

### **Code/**  
Complete Python implementation of the numerical models supporting the theoretical framework.

- **substrate_spectral_tests.py**  
  Full spectral‑geometry pipeline used in the paper.  
  Performs 1D finite‑size scaling, 2D Laplacian diagonalisation, degeneracy tests, coordinate‑chart correlations, graviton dispersion, and spectral‑dimension flow.

- **eigenmodes.py**  
  Builds the Gaussian relational entanglement kernel and diagonalises the low‑lying Laplacian spectrum to produce the `eigenmodes.png` figure used in the manuscript.

---

## 🌌 Core Theoretical Architecture

- **Structural Asymmetry:**  
  QFT fundamentally requires a fixed kinematic background to define field operator algebras.  
  GR denies a prior container, rendering the metric dynamical.  
  This architectural mismatch blocks conventional perturbative quantization.

- **The Baseline Principle:**  
  Gravity is the universal framework that makes field‑theoretic relations definable.  
  It cannot be quantized as a peer gauge field without destroying background independence.

- **Spectral Manifold Emergence:**  
  Space and metric tensors co‑emerge from the entanglement spectrum of a relational density matrix.  
  Modifying the underlying quantum information mapping is macroscopically equivalent to modifying the metric tensor \( g_{\mu\nu} \).

---

## 🚀 Execution and Local Compilation

### **Compiling the TeX Source Locally**
The manuscript is written using the official American Physical Society REVTeX 4‑2 document class.  
To compile the source into the final production PDF locally, ensure the image files are kept in the same directory as the source and run:

```bash
cd Manuscript
pdflatex main.tex
