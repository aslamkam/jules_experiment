# Predicting Amine pKa using Novel Fusion Models
## A Look at GCN-MLP and GCN-XGBoost

**Kamal Aslam**
University of Guelph

Ontario Graduate Mathematics Conference

---
# Introduction

* **What is pKa?**
    * Measures how acidic a compound is.
    * For amines (our focus), pKa indicates how easily they accept a proton – key to their chemical behavior.
* **Why is pKa important?**
    * **Drug Discovery:** Affects drug absorption, distribution, metabolism, excretion (ADME), and target interaction.
    * **Materials Science:** Influences reactivity and stability of materials.
* *Visual Cue: e.g., R-NH2 + H+ -> R-NH3+*
 
  `![Protonated Amine](https://ch302.cm.utexas.edu/images302/Dimethylammonium-formation-2D.png)
* **The Challenge:**
    * Lab measurement of pKa is accurate but can be slow and expensive for many molecules.
* **Our Goal:**
    * Build computational models for fast and accurate pKa prediction.
* **Focus Today:**
    * Two novel fusion models: GCN-MLP and GCN-XGBoost, which combine Graph Convolutional Networks with traditional methods.

---
# Machine Learning for pKa Prediction: Core Components

* **Traditional Machine Learning Methods:**
    * Examples: Multi-Layer Perceptrons (MLPs), Random Forests, XGBoost.
    * Typically use pre-defined numerical descriptions of molecules (descriptors/fingerprints).
* **Graph-Based Learning (GCNs):**
    * Graph Convolutional Networks learn directly from the molecule's 2D/3D structure.
    * Atoms = nodes; Bonds = edges.
    * GCNs can identify important features automatically, reducing manual feature engineering.
* **Today's Focus:**
    * Mathematical ideas behind combining GCNs with MLPs and XGBoost.

---
# Component 1: Graph Convolutional Networks (GCNs) - Basics

* **GCNs view molecules as graphs:**
    * **Nodes:** Atoms
    * **Edges:** Chemical bonds
* *Visual Cue:Diagram of Ethanol molecule as a graph*
  ![Ethanol Graph](https://newenergyandfuel.com/wp-content/uploads/2010/12/Ethanol-Molecule-in-3D.png))`
  
  *Example: Ethanol (CH₃CH₂OH) - Carbons and Oxygen are nodes, bonds are edges.*
* **Initial Node Features ($h_i^{(0)}$):**
    * Each atom `i` starts with a feature vector $h_i^{(0)}$.
    * In this work: 8 basic atomic properties (atomic number, charge, connectivity, hydrogen count, etc.) + 1 calculated partial charge.
    * Total: 9 initial features per atom.
* **Initial Feature Matrix ($H^{(0)}$):**
    * For a molecule with $N$ atoms, $H^{(0)}$ is an $N \times 9$ matrix.
    * Rows = atoms, Columns = features.

---
# Component 1: GCNs - How a GCN Layer Works

* **Purpose:** Update atom features by aggregating information from directly bonded neighbors.
* **Mathematical Rule (Conceptual):**
    $$H^{(l+1)} = \sigma( \text{Normalized\_Adjacency} \cdot H^{(l)} \cdot W^{(l)} )$$
* **Breaking it down:**
    * $H^{(l)}$: Feature matrix from the previous layer (or initial features for $l=0$).
    * $\text{Normalized\_Adjacency}$: (e.g., using $\tilde{A} = A + I_N$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$) Encodes connectivity and normalizes neighbor information to prevent scaling issues.
    * $W^{(l)}$: Learnable weight matrix for layer $l$. Transforms/re-weights features for the prediction task.
    * $\sigma$: Activation function (e.g., ReLU: $\max(0, x)$). Adds non-linearity.
* **In simpler terms:** Each atom aggregates transformed features from its neighbors, allowing local chemical environment information to be encoded into its updated feature vector $H^{(l+1)}$.

---
# Component 1: GCNs - Architecture and Output

* **Stacked GCN Layers:**
    * Typically multiple layers (e.g., 3 in this thesis).
    * $H^{(0)} \rightarrow \text{Layer 1} \rightarrow H^{(1)} \rightarrow \text{Layer 2} \rightarrow H^{(2)} \rightarrow \dots$
* **Common Techniques within each layer:**
    * **Batch Normalization:** Stabilizes learning.
    * **ReLU Activation:** Introduces non-linearity ($\max(0,x)$).
    * **Dropout:** Prevents overfitting by randomly dropping neuron outputs during training.
* **From Atom Features to Molecular Representation:**
    * After the final GCN layer, each atom has a rich feature vector.
    * **Global Pooling Layer** is needed for a single molecular representation.
        * Common method: **Global Mean Pooling** – average the final feature vectors of all atoms.
* **Molecular Embedding ($z_{GCN}$):**
    * The result of pooling (e.g., $z_{GCN}$).
    * A learned vector capturing key structural/chemical properties of the entire molecule, optimized for pKa prediction.

---
# Component 2: Multi-Layer Perceptrons (MLPs)

* **MLP Basics:**
    * A type of feedforward neural network.
    * Consists of an input layer, one or more hidden layers, and an output layer.
* *Visual cue: Simple Neuron or MLP structure diagram*
  ![Simple Neuron Diagram](https://iq.opengenus.org/content/images/2020/12/multilayer-perceptron-1.png)`
* **Neuron Operation:**
    * `output = activation_function( (Σ (inputs * weights)) + bias )`
* **MLP for pKa Prediction (Regression):**
    * **Input Layer:** Receives the feature vector (in fusion models, $z_{combined}$).
    * **Hidden Layers:**
        * Multiple neurons.
        * Use activation functions like ReLU ($\max(0,x)$) to learn non-linear patterns.
    * **Output Layer:**
        * Typically one neuron for pKa prediction.
        * Linear activation (outputs the raw sum).
* **Role in Fusion Models:** The MLP acts as a flexible learner, mapping combined features to the final pKa value.

---
# Component 3: XGBoost (Extreme Gradient Boosting)

* **XGBoost Overview:**
    * A powerful **ensemble learning** method, specifically **boosting**.
* **Boosting Principle:**
    * Builds a series of decision trees sequentially.
    * Each new tree corrects errors made by the previous ensemble of trees.
    * Prediction update: $F_m(x) = F_{m-1}(x) + \alpha \cdot h_m(x)$
        * $F_m(x)$: Prediction of model with $m$ trees.
        * $h_m(x)$: The new tree, trained on residuals of $F_{m-1}(x)$.
        * $\alpha$: Learning rate, controlling step size.
* **Decision Trees:** Split data into regions based on feature values, making simple predictions per region.
* **XGBoost Advantages:**
    * **Regularization:** Prevents overfitting.
    * **Efficiency:** Fast tree construction.
* **Role in Fusion Models:** Takes combined molecular features (including GCN embeddings) and uses its ensemble of trees for robust pKa prediction.

---
# Fusion Architecture: Combining Strengths

* **Rationale:**
    * GCNs excel at learning from graph structure (local atomic environments).
    * Other features (e.g., Benson groups, Sigma profiles) provide complementary global or expert-defined information.
* *Visual cue: Diagram showing the fusion concept (GCN branch + Aux features branch -> Combine -> Final Predictor)*
  `![Fusion Architecture Diagram](show Miro)`
* **Two-Branch Approach:**
    1.  **GCN Feature Extraction:**
        * Molecule (graph) $\rightarrow$ GCN $\rightarrow$ Molecular embedding $z_{GCN}$.
    2.  **Additional Feature Preparation:**
        * Calculate auxiliary features (e.g., Benson group counts, Sigma profiles) $\rightarrow x_{aux}$.
    3.  **Feature Combination:**
        * Concatenate: $z_{combined} = \text{concat}(z_{GCN}, x_{aux})$.
        * If $z_{GCN} = [g_1, \dots, g_P]$ and $x_{aux} = [a_1, \dots, a_Q]$, then $z_{combined} = [g_1, \dots, g_P, a_1, \dots, a_Q]$.
* **Input to Final Model:** $z_{combined}$ is fed into the MLP or XGBoost.

---
# Fusion Model 1: GCN-MLP

* **Concept:** Combines GCN-derived embeddings with auxiliary features, then uses an MLP for prediction.
* *Visual cue: GCN-MLP specific architecture diagram*
  `![GCN-MLP Diagram](show MIRO: gcn_mlp_diagram)`
    * GCN $\rightarrow z_{GCN}$
    * Auxiliary Features $\rightarrow x_{aux}$
    * $z_{combined} = \text{concat}(z_{GCN}, x_{aux})$
    * $z_{combined} \rightarrow$ MLP $\rightarrow$ Predicted pKa
* **MLP Structure:**
    * **Input Layer:** Size matches length of $z_{combined}$.
    * **Hidden Layers (1-3):** Neurons with ReLU activation ($\max(0,x)$).
    * **Output Layer:** Single neuron, linear activation.
* **Training:**
    * Can be end-to-end (GCN and MLP weights updated simultaneously).
    * Or, pre-train GCN, then use fixed $z_{GCN}$ to train MLP.

---
# Fusion Model 2: GCN-XGBoost

* **Concept:** Uses XGBoost as the final predictor on combined GCN embeddings and auxiliary features.
* *Visual cue: GCN-XGBoost specific architecture diagram*
  `![GCN-XGBoost Diagram](Show MIRO: gcn_xgboost_diagram)`
    * GCN $\rightarrow z_{GCN}$
    * Auxiliary Features $\rightarrow x_{aux}$
    * $z_{combined} = \text{concat}(z_{GCN}, x_{aux})$
    * $z_{combined} \rightarrow$ XGBoost $\rightarrow$ Predicted pKa
* **XGBoost Input:** The $z_{combined}$ vector.
* **XGBoost Operation:** Builds an ensemble of decision trees using features in $z_{combined}$.
* **Final Prediction:** Sum of predictions from all trees in the ensemble.
* **Training:**
    * Typically, pre-train GCN to get fixed $z_{GCN}$.
    * Then, use the static $z_{combined}$ dataset to train XGBoost.
    * End-to-end training is less common with XGBoost in this setup.

---
# Why Fusion Works: Conceptual Synergies

* **1. GCNs for 'Representation Learning':**
    * Automatically learn informative numerical summaries ($z_{GCN}$) from molecular structure.
    * Capture local chemical context effectively.
* **2. Auxiliary Features ($x_{aux}$) for 'Complementary Information':**
    * Benson groups: Known chemical fragments related to properties.
    * Sigma profiles: Information about molecular surface charge and polarity.
    * Capture information a GCN might not prioritize or might require many layers to learn.
* **3. Powerful Final Predictors (MLP & XGBoost):**
    * **MLP:** Finds complex, non-linear relationships in $z_{combined}$.
    * **XGBoost:** Robust with diverse data types and complex interactions; its tree structure is good at feature interaction.
* **Outcome:** The combined feature vector $z_{combined}$ provides a richer information set, leading to improved prediction accuracy.

---
# Key Results

* **Main Finding:** Hybrid models (GCN-MLP, GCN-XGBoost) generally showed higher accuracy for amine pKa prediction compared to standalone GCN, MLP, or XGBoost models.
* **Optimal GCN Input Features ('data G1'):**
    * 8 basic atomic features + Gasteiger partial charges per atom.
* **Clarification on "Aggregated":** GCNs use per-atom features. These are then aggregated (e.g., by averaging via global pooling) to get a single molecular vector $z_{GCN}$.
* **Best Fusion Performance:**
    * Often achieved by combining $z_{GCN}$ (from 'data G1') with Benson group features.
    * This combined vector was then fed into the MLP or XGBoost model.
* **Performance Metrics:**
    * **Mean Squared Error (MSE):** Lower is better.
    * **Coefficient of Determination ($R^2$):** Closer to 1 is better (indicates predictions align well with experimental values).
    * *(Optional: "For instance, $R^2$ might improve from 0.8 for a GCN alone to 0.85 for a GCN-XGBoost model, showing this combined benefit.")*

---
# Conclusion & Future Directions

* **Summary:**
    * Fusion models (GCN-MLP, GCN-XGBoost) are effective for predicting amine pKa.
    * They successfully merge automatically learned graph features with other informative chemical descriptors.
    * GCNs capture local atomic environments; this is enhanced by auxiliary features and powerful regression tools (MLP/XGBoost).
* **Main Takeaway:** Combining GCNs (for graph learning) with robust regression methods is a promising strategy for chemical property prediction.
* **Future Work:**
    * Explore **attention mechanisms** in GCNs (to focus on important molecular regions).
    * Investigate end-to-end training for GCN-XGBoost (challenging but potentially beneficial).
    * Incorporate a wider variety of molecular features.

---
# Acknowledgements

* **Co-advisors:** Dr. William R. Smith and Dr. Mihai Nica – for their invaluable guidance and support.
* **Computational Resources:**
    * `SHARCNET, Compute Canada, University Clusters

---
# Q&A

## Thank you for listening!

![Final Image](https://th.bing.com/th/id/R.6c8d5488ef173ac2b4e27860b1d1e2cb?rik=Z1PWZ6v0nW5Qag&riu=http%3a%2f%2fmedia.cntraveler.com%2fphotos%2f57212200afc531496352e35e%2fmaster%2fpass%2fgolden-gate-bridge-GettyImages-166267310.jpg&ehk=Wwmnq1LY6HE7nb%2f6ElFEtu%2biyHkBQedlLREuaP3NKAQ%3d&risl=&pid=ImgRaw&r=0)`