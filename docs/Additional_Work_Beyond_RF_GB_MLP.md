# Models Added Beyond RF, GB, and MLP

The project originally used the conventional baseline learners: Random Forest (RF), Gradient Boosting (GB), and Multi-Layer Perceptron (MLP). These models establish a standard data-driven benchmark for PMUT frequency-response prediction. However, the project extends beyond these tabular baselines by introducing physically structured models, multi-mode dynamical models, and hybrid residual-learning frameworks. These added models are not simply alternative regressors; they are designed to encode the resonance physics of PMUT devices and to provide more interpretable predictions of the underlying device behavior.

The added models are implemented in:

- [pinn_model.py](../src/Helpers/pinn_model.py)
- [mdof_pinn_model.py](../src/Helpers/mdof_pinn_model.py)
- [pinn_informed_loss_model.py](../src/Helpers/pinn_informed_loss_model.py)
- [rf_mdof_pinn_ensemble.py](../src/Helpers/rf_mdof_pinn_ensemble.py)
- [bvd_model.py](../src/Helpers/bvd_model.py)
- [mdof_model.py](../src/Helpers/mdof_model.py)

---

## 1. PINN

A Physics-Informed Neural Network (PINN) was introduced to enforce known dynamical structure in the prediction model. Instead of learning only a mapping from PMUT index and frequency to amplitude, the PINN predicts physically meaningful parameters such as resonant frequency $f_0$, damping coefficient $c$, stiffness $k$, gain $G$, and baseline offset $A_{bg}$. The predicted spectrum is then reconstructed using a reduced-order oscillator model.

The physics layer follows the form

$$
|X(\omega, N)| = \frac{\alpha_N V}{\sqrt{(k_N - m\omega^2)^2 + (c_N \omega)^2 + \epsilon}}
$$

and the output response is modeled as

$$
A_{pred}(f, N) = G_N |X(\omega, N)| + A_{bg}
$$

where $\omega = 2\pi f$, $m$ is the effective mass, $V$ is the excitation amplitude, and $\epsilon$ is a small numerical stabilizer.

This design is important because it transforms the task from a purely black-box regression problem into a structured surrogate model that embeds known resonance physics. The resulting model is physically interpretable and better suited for extrapolation across PMUT devices with similar dynamic behavior.

### Why it was included

The PINN was added to test whether a model constrained by physical equations can generalize better than purely data-driven baselines when only a small number of PMUTs are available for training. In a regime where the training set is small, embedding dynamic structure into the model is a stronger inductive bias than learning the entire response from raw samples alone.

---

## 2. MDOF-PINN

The MDOF-PINN extends the PINN approach to a multi-mode resonant system. Rather than assuming a single dominant mode, it models multiple resonant contributions simultaneously, each with its own amplitude $A_i$, resonant frequency $f_{0,i}$, and quality factor $Q_i$. This is a more realistic representation for PMUT frequency responses, which may contain several interacting resonance modes.

The model approximates the response using a superposition of modal transfer functions:

$$
H_i(\omega, N) = \frac{A_i(N)\,\omega_{0,i}(N)^2}{(\omega_{0,i}(N)^2 - \omega^2) + j\,\omega_{0,i}(N)\omega / Q_i(N)}
$$

and the predicted response is

$$
A_{pred}(f, N) = \left|\sum_{i=1}^{K} H_i(\omega, N)\right| + A_{bg}(N)
$$

where $K$ is the number of active modes.

This model is particularly relevant for PMUT arrays because resonant systems often contain multiple mechanical modes. The use of MDOF-PINN therefore aims to capture the richer spectral structure of the device and to improve the physical fidelity of the learned representation beyond the single-mode approximation.

### Why it was included

This model was added to represent multi-modal dynamic behavior that cannot be adequately captured by single-degree-of-freedom formulations. It was introduced to test whether explicitly modeling multiple resonant modes leads to better signal reconstruction and parameter estimation across a PMUT population.

---

## 3. PINN-Informed-Loss

The PINN-Informed-Loss model extends the physics-informed learning framework by including additional regularization terms directly in the training objective. In this formulation, the total loss is a weighted sum of data fidelity and physical consistency terms:

$$
\mathcal{L}_{total} = \lambda_d \mathcal{L}_{data} + \lambda_p \mathcal{L}_{physics} + \lambda_s \mathcal{L}_{smooth}
$$

where:

- $\mathcal{L}_{data}$ measures the mismatch between measured and predicted amplitudes,
- $\mathcal{L}_{physics}$ enforces physical plausibility of the estimated parameters,
- $\mathcal{L}_{smooth}$ penalizes excessive variation between neighboring PMUT conditions.

A typical data term is the mean-squared error:

$$
\mathcal{L}_{data} = \frac{1}{M} \sum_{i=1}^{M} (A_{pred,i} - A_{exp,i})^2
$$

and the physics term promotes positive and consistent parameters such as mass, stiffness, damping, and quality factor.

This approach was added to improve robustness, interpretability, and stability, particularly in cases where the data are limited or where the learned parameters may drift away from physically meaningful values during optimization.

### Why it was included

The informed-loss model was introduced as a stronger constrained-learning strategy: the network is trained not only to fit the measured response but also to respect physical constraints and smoothness across the PMUT array. This reduces the risk of producing nonphysical parameter combinations while maintaining predictive capability.

---

## 4. RF-MDOF-PINN Ensemble

The RF-MDOF-PINN ensemble combines a physics-informed backbone with a residual Random Forest correction. The base physics model learns the dominant resonant structure, while the RF model captures systematic residual errors that remain after the physics-based prediction.

The prediction can be written in the form

$$
A_{ensemble}(f, N) = A_{phys}(f, N) + r_{RF}(f, N)
$$

where $A_{phys}(f, N)$ is the MDOF-PINN prediction and $r_{RF}(f, N)$ is the residual estimated by the Random Forest.

This hybrid design is motivated by the fact that physics-driven models are strong at representing the underlying resonance structure, whereas data-driven models can compensate for deviations due to device variability, imperfect parameterization, or unmodeled effects.

### Why it was included

This model was added to test the benefit of a hybrid “physics + correction” strategy. Instead of relying completely on a fixed physics model or a completely unconstrained machine learner, the ensemble uses one model to capture the dominant structure and another to model remaining discrepancies. This is valuable in settings where the physical model is informative but not exact.

---

## 5. BVD Equivalent-Circuit Model

The Butterworth–Van Dyke (BVD) model is a classical equivalent-circuit model used to represent resonant piezoelectric devices. It describes the PMUT response in terms of electrical components representing motional stiffness, damping, capacitance, and parasitic effects.

The model is generally used to fit the electrical admittance or impedance response and infer physically meaningful circuit parameters. In this project, the BVD formulation is used as a benchmark physics-based description of the PMUT resonance and provides an interpretable alternative to purely neural surrogates.

This formulation is implemented in [bvd_model.py](../src/Helpers/bvd_model.py), where the model is fit directly to the PMUT frequency-response data to estimate parameter values associated with resonance and electromechanical coupling.

### Why it was included

The BVD model was added to provide a classical, well-established physical baseline against which more recent physics-informed and hybrid methods could be compared. It is useful for interpreting the PMUT response in conventional circuit terms and for benchmarking whether learned models capture physically meaningful trends.

---

## 6. MDOF Mechanical Model

The Multi-Degree-of-Freedom (MDOF) mechanical model decomposes the PMUT response into a sum of several mechanical resonant contributions. Physically, this corresponds to representing the device as a system with multiple vibrational modes, each characterized by its own modal amplitude, resonant frequency, and damping.

In compact form, the response can be represented as

$$
A_{pred}(f) = A_{bg} + \sum_{i=1}^{K} A_i \cdot \Phi_i(f; f_{0,i}, Q_i)
$$

where $\Phi_i$ is the modal transfer function associated with mode $i$, and $A_{bg}$ is the baseline offset.

This model is important because it captures the fact that PMUT responses are not necessarily governed by a single resonance. The implementation in [mdof_model.py](../src/Helpers/mdof_model.py) allows the estimation of modal parameters rather than only a single overall response curve.

### Why it was included

This model was added to characterize the governing mechanical structure of the PMUT array more explicitly. It provides an interpretable and physically grounded alternative to purely statistical learning and supports direct comparison with PINN-based and MDOF-PINN formulations.

---

## Results and evaluation logic

The project evaluates these models using the same core metrics used throughout the benchmarking workflow: MAE and $R^2$.

The prediction error is measured as

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

and the coefficient of determination is

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

These metrics are computed for both the full spectrum and the FWHM window, which isolates the resonant peak region. The project also distinguishes between progressive training and regime-aware training schemes, allowing evaluation of model generalization as the training set grows and as the target PMUT moves across different operating regimes.

### Interpretation of the results

The project explicitly documents that negative $R^2$ values indicate a model performs worse than simply predicting the mean response. This is expected in low-data or highly shifted regimes, where a learned model is extrapolating far from its training distribution. In the project workflow, these failures are not treated as arbitrary noise; they are used to diagnose the limits of each model, especially in early progressive steps or cross-regime prediction tasks.

The main outputs produced by the project are saved in the generated metrics and HTML comparison files, including:

- [progressive_metrics.csv](../results/metrics/progressive_metrics.csv)
- [regime_metrics.csv](../results/metrics/regime_metrics.csv)
- [progressive explorer](../supplementary/explorer/all_combined.html)
- [regime explorer](../supplementary/explorer/regime/all_combined.html)
- [BVD parameter fits](../results/direct_fits/bvd_parameters_comparison.html)
- [MDOF parameter fits](../results/direct_fits/mdof_parameters_comparison.html)
- [PINN physics comparison](../results/direct_fits/pinn_physics_comparison.html)

### Overall conclusion

The additional models are not redundant extensions of RF, GB, and MLP; they are a methodological shift toward physics-aware, interpretable, and multi-modal PMUT modeling. The combination of constrained neural models, equivalent-circuit fitting, and hybrid residual learning provides a richer and more scientifically grounded framework for PMUT characterization than classical black-box regressors alone.

This set of models represents the main methodological advancement of the project beyond the initial baseline trio, and it is precisely this combination of physical modeling and data-driven learning that makes the framework suitable for publication-quality analysis in a PMUT modeling study.
