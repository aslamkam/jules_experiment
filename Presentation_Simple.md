# Presentation Script: Predicting Amine pKa using Novel Fusion Models

## Slide 1: Title Slide

*   **Script:** "Good morning/afternoon, everyone. My name is Kamal Aslam, from the University of Guelph. Today, I'll be presenting a part of my thesis research titled 'Predicting Amine pKa using Novel Fusion Models: A Look at GCN-MLP and GCN-XGBoost.' This work was prepared for the Ontario Graduate Mathematics Conference."

## Slide 2: Introduction

*   **Script:**
    *   "Let's start with pKa. Simply put, pKa measures how acidic a compound is. For amines, the molecules we're studying, pKa tells us how easily they accept a proton, which is key to their chemical behavior."
    *   "Why is this important? In **drug discovery**, pKa affects how a drug moves through the body and how it interacts with its target. In **materials science**, pKa influences a material's reactivity and stability."
    *   *(Presenter gestures to an image of a protonated amine).*
    *   "The challenge is that measuring pKa in the lab, while accurate, can be slow and expensive, especially if you have many molecules to test."
    *   "So, our goal was to build computational models that can predict these pKa values quickly and accurately, offering a useful alternative to lab experiments."
    *   "In my thesis, I looked at various machine learning models. Today, I'll focus on two new 'fusion' models that combine Graph Convolutional Networks (GCNs) with more traditional methods – specifically GCN-MLP and GCN-XGBoost – as these showed very good results."

## Slide 3: Machine Learning for pKa Prediction: Core Components

*   **Script:**
    *   "Before we get to the fusion models, let's briefly cover the building blocks. We explored several **traditional machine learning methods** like Multi-Layer Perceptrons (MLPs), Random Forests, and XGBoost. These models usually work with pre-defined numerical descriptions of molecules, often called 'descriptors' or 'fingerprints'."
    *   "More recently, **graph-based learning** has become popular. Graph Convolutional Networks, or GCNs, are exciting because they learn directly from the molecule's structure – thinking of atoms as points (nodes) and bonds as connections (edges). This means the GCN can figure out important features on its own, reducing the need for us to define them manually."
    *   "Today, we'll look at the **mathematical ideas** behind how we combined these GCNs with MLPs and XGBoost to create effective prediction tools."

## Slide 4: Component 1: Graph Convolutional Networks (GCNs) - Basics

*   **Script:**
    *   "So, how do GCNs see molecules? They see them as graphs. The **atoms** are the **nodes** in the graph, and the **chemical bonds** are the **edges** connecting these nodes."
    *   *(Presenter refers to a diagram, e.g., of ethanol).* "For example, in a simple molecule like ethanol, each carbon and oxygen atom is a node, and the lines representing bonds are the edges."
    *   "For a GCN to work, each atom (or node) starts with a set of initial numbers that describe it – these are its features. We can call the feature vector for atom `i` as `h_i^(0)` (h-i-super-zero)."
    *   "In my work, these initial features included basic atomic properties like atomic number, charge, how many other atoms it's connected to, how many hydrogens it has, and so on. We used 8 such properties, plus one more for the atom's calculated partial charge. This gave us a starting list of 9 numbers for each atom."
    *   "So, each atom `i` begins with this 9-number vector `h_i^(0)`."
    *   "If we collect these feature vectors for all `N` atoms in a molecule, we get a starting matrix `H^(0)` (H-super-zero). This matrix has `N` rows (one for each atom) and 9 columns (one for each feature)."

## Slide 5: Component 1: GCNs - How a GCN Layer Works

*   **Script:**
    *   "The main job of a GCN layer is to update the features of each atom by looking at its directly bonded neighbors. It’s like each atom gets a better description by considering its local environment."
    *   "This update happens using a mathematical rule. A common way to write this is:
        `H^(l+1) = σ( Normalized_Adjacency * H^(l) * W^(l) )`."
    *   "Let's unpack this conceptually:
        *   We start with `H^(l)`, which are the feature numbers for all atoms from the previous step (or the initial features if it's the first step).
        *   The `Normalized_Adjacency` part (which involves terms like `Ã` and `D̃` you might see in papers) essentially tells the GCN how atoms are connected and helps in 'averaging' or 'normalizing' the information gathered from neighbors. This stops the numbers from getting too big or too small.
        *   This averaged neighbor information is then multiplied by `W^(l)`, a 'learnable weight matrix'. Think of `W^(l)` as a set of parameters that the GCN learns during training. It transforms or re-weights the features to make them more useful for the prediction task.
        *   Finally, `σ` (sigma) is an 'activation function,' often ReLU, which just means `max(0, x)`. This step adds non-linearity, which is important because molecular properties often have non-linear relationships with their structures."
    *   "So, in simpler terms: each atom looks at its neighbors' features, averages them in a weighted way, transforms them using what it has learned (`W^(l)`), and applies a simple non-linear tweak (`σ`). This process allows information about the local chemical environment to be encoded into each atom's updated feature vector `H^(l+1)`."

## Slide 6: Component 1: GCNs - Architecture and Output

*   **Script:**
    *   "A GCN model is typically made by stacking several of these GCN layers. In my thesis, we often used three such layers."
    *   "The initial features `H^(0)` go into the first layer, get updated to `H^(1)`, then `H^(1)` goes into the second layer to become `H^(2)`, and so on."
    *   "Inside each layer, we also use a few standard techniques:
        *   **Batch Normalization:** This helps make the learning process more stable.
        *   **ReLU Activation:** The `max(0,x)` function we just talked about.
        *   **Dropout:** This is a method to prevent the model from 'memorizing' the training data too well (overfitting), by randomly ignoring some neuron outputs during training."
    *   "After the last GCN layer, each atom has a new feature vector that's rich with information learned from its surroundings."
    *   "But we need one overall representation for the whole molecule to predict a single pKa value. So, we use a **Global Pooling Layer**."
    *   "A common method is Global Mean Pooling: we simply take the average of the final feature vectors of all atoms in the molecule. This gives us one vector, `z_GCN`."
    *   "This `z_GCN` vector is the **molecular embedding**. It’s a learned set of numbers that captures the important structural and chemical properties of the whole molecule, tailored for predicting pKa."

## Slide 7: Component 2: Multi-Layer Perceptrons (MLPs)

*   **Script:**
    *   "The next ingredient in our fusion models is the Multi-Layer Perceptron, or MLP. This is a well-known type of neural network."
    *   "An MLP has an input layer, one or more hidden layers, and an output layer. Data flows in one direction: from input, through the hidden layers, to the output."
    *   "Each 'neuron' in an MLP takes its inputs, multiplies them by weights, adds a bias, and then passes this sum through an activation function. The formula is roughly `output = activation_function( (Σ (inputs * weights)) + bias )`."
    *   *(Presenter refers to a simple neuron diagram).*
    *   "For predicting a value like pKa (which is a regression task):
        *   The **Input Layer** takes the feature vector – in our fusion models, this will be a combined set of features, `z_combined`.
        *   **Hidden Layers** have several neurons. They use an activation function like ReLU (`max(0,x)`) to help the MLP learn non-linear patterns in the data.
        *   The **Output Layer** for pKa prediction typically has just one neuron. Since pKa is a number, this neuron usually has a linear activation, meaning it just outputs the calculated sum."
    *   "In our fusion models, the **MLP acts as a flexible tool** that takes the combined features (from the GCN and other sources) and learns how to map them to the final pKa value."

## Slide 8: Component 3: XGBoost (Extreme Gradient Boosting)

*   **Script:**
    *   "The third component, used in our GCN-XGBoost fusion, is XGBoost, which stands for Extreme Gradient Boosting."
    *   "XGBoost is a powerful type of **ensemble learning**, specifically **boosting**. The main idea of boosting is to build a series of models (in this case, decision trees) one after another. Each new tree tries to correct the mistakes made by the previous set of trees."
    *   "Think of it like this: the overall prediction of the model `F_m(x)` after `m` trees have been added is `F_m(x) = F_(m-1)(x) + α * h_m(x)`. This means the **new model's prediction is the old model's prediction plus a small improvement (`α * h_m(x)`) from the new tree** `h_m(x)`."
    *   "This new tree `h_m(x)` is specifically trained to fix the errors (or 'residuals') of the previous model `F_(m-1)(x)`. The term `α` (alpha) is a learning rate that controls how big of an improvement step we take with each new tree."
    *   "These **Decision Trees** work by splitting the data into different regions based on feature values, and then making a simple prediction for each region."
    *   "XGBoost is particularly good because it includes:
        *   **Regularization**, which helps prevent it from becoming too complex and overfitting the training data.
        *   It's also very **efficient** at building these trees."
    *   "In our fusion setup, XGBoost takes the combined molecular features (including those from the GCN) and uses its ensemble of trees to make a robust pKa prediction."

## Slide 9: Fusion Architecture: Combining Strengths

*   **Script:**
    *   "So, how do we put these pieces together? Our idea was that GCNs are good at learning from the molecule's graph structure, understanding the local environment of each atom. But other types of information, like the presence of specific chemical groups (Benson groups) or overall electronic properties (Sigma profiles), could add even more value."
    *   *(Presenter gestures to a diagram showing the fusion concept).* "Our fusion models generally follow a two-branch approach that feeds into a final prediction model."
    *   "**Step 1 is GCN Feature Extraction:**
        *   The molecule (as a graph with initial atom features) goes into the GCN.
        *   The GCN processes it, as we discussed.
        *   The output is a learned 'molecular embedding' – a vector of numbers `z_GCN` that summarizes the molecule from the GCN's perspective."
    *   "**Step 2 involves Additional Feature Preparation:**
        *   We also calculate other features for the molecule, like Benson group counts or Sigma profiles.
        *   These features, let's call them `x_aux`, give us different kinds of information not directly captured by the GCN in the same way."
    *   "**Step 3 is Feature Combination:**
        *   We then simply combine these two sets of features by concatenating them: `z_combined = concat(z_GCN, x_aux)`."
        *   "Mathematically, if `z_GCN` is a list of numbers `[g_1, ..., g_P]` and `x_aux` is `[a_1, ..., a_Q]`, then `z_combined` is just a longer list `[g_1, ..., g_P, a_1, ..., a_Q]`."
    *   "This `z_combined` vector, which contains information from both the GCN and our other chosen features, becomes the input for our final MLP or XGBoost model."

## Slide 10: Fusion Model 1: GCN-MLP

*   **Script:**
    *   "Our first fusion model, GCN-MLP, combines the GCN with a Multi-Layer Perceptron."
    *   *(Presenter points to the GCN-MLP diagram).* "You can see the GCN produces its molecular embedding `z_GCN`. Separately, other features `x_aux` are prepared. They are joined to form `z_combined`, which is then fed into an MLP to predict the pKa."
    *   "The **input to this MLP** is that `z_combined` vector."
    *   "A **typical MLP structure** for this task would have:
        *   An **Input Layer** with a number of neurons matching the length of `z_combined`.
        *   One to three **Hidden Layers**, where neurons use the ReLU activation function (`max(0,x)`). The number of neurons in these layers is something we fine-tune. The first hidden layer takes `z_combined` as input, and subsequent layers take the output of the previous hidden layer.
        *   An **Output Layer** with a single neuron (and no complex activation) to give us the final pKa number."
    *   "For **training**, the GCN and MLP components can be trained 'end-to-end.' This means the prediction error is used to adjust the learnable weights in both the MLP and the GCN simultaneously. Alternatively, the GCN can be pre-trained first, and its learned embeddings then used as fixed inputs to train the MLP."

## Slide 11: Fusion Model 2: GCN-XGBoost

*   **Script:**
    *   "Our second model, GCN-XGBoost, uses XGBoost as the final predictor."
    *   *(Presenter points to the GCN-XGBoost diagram).* "Similar to the GCN-MLP, the GCN generates `z_GCN`, which is combined with `x_aux` to create `z_combined`. This `z_combined` vector is then used to train the XGBoost model."
    *   "The **input to XGBoost** is `z_combined`."
    *   "**XGBoost then builds an ensemble of decision trees.** Each tree uses the features in `z_combined` (some from the GCN, some from `x_aux`) to make decisions and predict parts of the pKa value (specifically, the errors of previous trees)."
    *   "Conceptually, the **final pKa prediction** is a sum of the predictions from all the trees in the XGBoost ensemble."
    *   "For **training** GCN-XGBoost, it's more common to pre-train the GCN first to get fixed `z_GCN` embeddings. These are then combined with `x_aux`, and this static `z_combined` dataset is used to train the XGBoost model. Training both parts together from scratch is less typical for XGBoost."

## Slide 12: Why Fusion Works: Conceptual Synergies

*   **Script:**
    *   "So, why do these fusion models often outperform individual models? It's about combining different strengths."
    *   "First, **GCNs are great at 'Representation Learning.'** They automatically learn useful numerical summaries (`z_GCN`) from the molecule's structure. The GCN layers effectively spread information between neighboring atoms, capturing the local chemical context without us having to manually design all these features."
    *   "Second, the **other features (`x_aux`) provide 'Complementary Information.'** For example, Benson groups are well-known chemical fragments that chemists already know are related to certain properties. Sigma profiles tell us about the molecule's surface charge and polarity. These can capture information that a GCN might not focus on, or might take many layers to learn."
    *   "Third, MLPs and XGBoost are **Powerful Final Predictors:**
        *   An **MLP** can find complex, non-linear patterns between all the combined features in `z_combined` and the pKa value.
        *   **XGBoost** is also very good with various types of data and complex relationships. Its tree-based structure is effective at finding how different features interact."
    *   "In short, the **combined feature vector `z_combined` gives the final model a richer set of information** than either the GCN embeddings alone or the auxiliary features alone, leading to better predictions."

## Slide 13: Key Results (Brief Overview from Thesis Abstract)

*   **Script:**
    *   "Let's briefly look at what our research found, as highlighted in the thesis."
    *   "A key finding was that these **hybrid models, GCN-MLP and GCN-XGBoost, generally predicted amine pKa values more accurately** than when we used GCNs, MLPs, or XGBoost on their own."
    *   "For the **GCN part**, it worked best when using 8 basic atomic features plus Gasteiger partial charges for each atom (this set was called 'data G1')."
    *   *(Self-correction clarification for the audience):* "Just to clarify, GCNs use features for each atom. The term 'aggregated' that might appear in some descriptions refers to a final step where these processed atom features are combined (e.g., by averaging) to get one vector for the whole molecule, `z_GCN`."
    *   "For the **fusion models**, the best results often came from combining the GCN's molecular embedding (`z_GCN` from 'data G1') with Benson group features, and then feeding this into the MLP or XGBoost model."
    *   "We measured these improvements using standard statistics like **Mean Squared Error (MSE)** – which we want to be small – and the **coefficient of determination (R²)** – which we want to be close to 1, indicating our predictions closely match experimental values."
    *   *(Optional: "For instance, R² might improve from 0.8 for a GCN alone to 0.85 for a GCN-XGBoost model, showing this combined benefit.")*

## Slide 14: Conclusion & Future Directions

*   **Script:**
    *   "In **summary**, our work shows that fusion models like GCN-MLP and GCN-XGBoost are effective for predicting amine pKa values. They successfully combine automatically learned graph features with other informative chemical descriptors."
    *   "GCNs are good at understanding the local atomic environments within molecules. When we combine these learned insights with other features, and use powerful tools like MLPs or XGBoost to make the final prediction, we can achieve better accuracy."
    *   "The **main takeaway** is that combining the strengths of GCNs (for learning from molecular graphs) with robust regression methods (like MLPs and XGBoost) is a promising approach for predicting chemical properties from molecular structure."
    *   "For **future work**, one could explore:
        *   Using 'attention mechanisms' in GCNs, which might help the model focus on more important parts of the molecule.
        *   Trying to train the GCN and XGBoost parts together from scratch, which can be challenging but might offer benefits.
        *   Adding even more types of molecular features."

## Slide 15: Acknowledgements

*   **Script:** "Before I finish, I'd like to express my sincere gratitude to my co-advisors, Dr. William R. Smith and Dr. Mihai Nica, for their guidance and support."
*   \[SHARCNET, Compute Canada, if applicable]

## Slide 16: Q&A

*   **Script:** "Thank you for listening. I'd be happy to answer any questions."
*   *(Presenter might refer to a final, impactful image on the slide).*