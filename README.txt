THE WARM ICING MODEL
Quantum-assisted Bayesian calibration of chip-package warpage

Infers which regions of a package substrate are copper-rich from a measured warpage field,
with calibrated uncertainty. The posterior over copper maps is exactly an Ising model and is
sampled with quantum-enhanced MCMC proposals (Layden et al., Nature 619, 282 (2023)). The
proposal circuit was run on IonQ, IBM and Rigetti hardware and on vendor noise models.

Qollab × IonQ hackathon 2026, theme Quantum Impact. MIT license.
This effort is supported by Qollab and IonQ.


START HERE
  quantum_runs.xlsx      every quantum run: device, date, job ID, shots, results, raw counts file
  qollab/                the project page (run the circuit, see how each number is calculated)
  docs/                  where every input came from, what was synthetic, what still needs checking
  Everything else is the code and data behind those numbers, kept so any result can be re-run.


RUN
  On Qollab: upload the contents of qollab/ as a project, press Run and choose a backend, then pick
             the 9- or 12-qubit circuit on the page and press "Run on selected backend".
  Locally:   pip install -r requirements-local.txt
             python -c "import main; from qiskit_aer import AerSimulator; main.backend = AerSimulator(); main.main()"

VERIFY
  python verify_all.py --specs     recomputes every hardware number from the raw counts files
  RESULTS_MAP.txt                  every reported number, the script and file behind it, how to re-run it

LAYOUT
  qollab/                                  project page for Qollab (HTML, CSS, JavaScript only)
  docs/                                    data provenance and fact-check guide (Word and PDF)
  main.py                                  the full pipeline as one script
  benchmark/                               3D FEA, published-measurement check, synthetic benchmark,
                                           calibration, stress tests, classical baselines
  hardware_9q/, hardware_12q/              circuits, run and scoring scripts, one counts file per run

RESULTS IN BRIEF
  Forward physics: -59.2 vs -58 um at 25 C on the package of Tsai, Wang & Liu (2021); 3.2 um RMS
  over 25-260 C. Calibration lowers warpage prediction error 2.8-15x on synthetic boards.
  9-qubit circuit: every IonQ run (XEB 0.46-0.96) above every IBM run (0.10-0.20); Rigetti 0.03.
  12-qubit circuit: IonQ Forte Enterprise 1 0.49 +/- 0.12, IBM Kingston 0.43 +/- 0.03, Rigetti 0.02.

LIMITATIONS
  No quantum advantage is claimed: enumeration is exact at 9-12 regions and classical block-move
  Metropolis matches the quantum proposal at 3% noise. Speed-ups are modelled; the full sampler was
  not run on hardware. Benchmark boards are synthetic; only the forward physics is checked against a
  published measurement. Hardware runs: 1-3 per device.


CREDITS
  Method: Layden et al., Quantum-enhanced Markov chain Monte Carlo, Nature 619, 282 (2023).
  Measured data and materials: Tsai, Wang & Liu, Materials 14:3723 (2021), Fig. 13 and Table 1 (CC BY 4.0).
  Geometry: Cadence warpage white paper. Measurement precision: Kulterman et al., iNEMI, IEEE IEMT (2016).
  Full source list and what was taken from each: docs/Warm_Icing_Model_data_provenance.pdf
  Software: Qiskit, Qiskit Aer, qiskit-ionq, qiskit-ibm-runtime, scikit-fem, NumPy, SciPy, pandas,
  Matplotlib, dwave-samplers. Hardware access: Qollab, qBraid, IBM Quantum.


COMPUTE AND COSTS
  qBraid: IonQ Forte Enterprise 1 via AWS (830 credits), Rigetti Cepheus 9q (60.27) and 12q (65.88),
          AWS Braket DM1 (0.57). Total 956.72 credits ($9.57 at $0.01 per credit).
  Qollab: IonQ Forte 1, two runs (78.34 credits together) and IonQ Forte Enterprise 1
          (25.79 credits); 104.13 credits in total. IonQ noise-model simulator runs were free.
  OpenQuantum (via qBraid): IonQ Forte 1, run 3, free tier.
  IBM Quantum Open Plan: 5 jobs, 28 s of QPU time, free.
  Every job with its ID and cost: quantum_runs.xlsx

THANK YOU
  Thank you to the quantum hardware and platform providers for the access and credits that made these
  experiments possible. Being able to run the circuits on real devices and compare them with
  simulations made a big difference to what could be explored during the hackathon.


AI ASSISTANCE DISCLOSURE
  This project was developed during the October 9-11, 2026 hackathon. Shivani Mayekar chose the
  problem, reviewed the literature, and led the research direction and experimental planning. She ran
  the hardware experiments, set the evaluation priorities, and made the final reporting decisions.
  Claude (Anthropic) was used extensively throughout development. It generated most of the code and
  contributed to the mathematical formulation, technical analysis, documentation, and website
  implementation.
  The repository includes the results, validation checks, limitations, and supporting data. The saved
  measurement counts and analysis scripts can be used to recompute the reported hardware metrics
  (python verify_all.py --specs).
