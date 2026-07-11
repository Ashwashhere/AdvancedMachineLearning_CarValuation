<div align="center">
  <h1>🚗 Advanced Machine Learning: UK Used Car Valuation</h1>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.13+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/Jupyter-Notebook-orange.svg" alt="Jupyter" />
    <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-lightgrey.svg" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-yellow.svg" alt="SHAP" />
    <img src="https://img.shields.io/badge/SciPy-Advanced%20Math-8CAAE6.svg" alt="SciPy" />
    <img src="https://img.shields.io/badge/Pandas-Data%20Processing-green.svg" alt="Pandas" />
  </p>
</div>

<hr />

<h2>📌 Project Overview</h2>
<p>
  The used car market is highly dynamic, with vehicle depreciation driven by a complex combination of age, mileage, brand, and market segmentation. This project designs and deploys an end-to-end <b>Advanced Machine Learning architecture</b> to optimise vehicle valuation tracking over a comprehensive dataset of ~402,000 UK used car listings.
</p>
<p>
  This framework moves beyond standard regressors by introducing <b>Meta-Ensemble Stacked Architectures</b>, linear and non-linear manifold learning (<b>PCA, t-SNE, Isomap</b>), and a multi-algorithm clustering pipeline (<b>K-Means, DBSCAN, Hierarchical Ward's Linkage</b>) applied to <b>SHAP (Explainable AI)</b> dependency values. The result is a highly interpretable, production-grade valuation pipeline that maps out latent market dynamics while minimising prediction error.
</p>

<hr />

<h2>📊 Dataset & Architecture Summary</h2>
<ul>
  <li><b>Total Volumetric Scope:</b> ~402,000 source observations.</li>
  <li><b>Rigorous Data Partitioning:</b> 60% Training, 20% Validation, 20% Test split structured strictly prior to any downstream transformations to ensure absolute insulation against data leakage.</li>
  <li><b>Variance Stabilisation:</b> Applied a log-transformation (<code>np.log1p</code>) to the highly right-skewed <code>price</code> target variable.</li>
  <li><b>Feature Scope:</b> <code>mileage</code>, <code>standard_make</code>, <code>standard_model</code>, <code>vehicle_condition</code>, <code>body_type</code>, <code>fuel_type</code>, and <code>reg_code</code>.</li>
</ul>

<hr />

<h2>⚙️ Methodology & Pipeline Execution</h2>

<h3>1. Robust Preprocessing & Feature Engineering</h3>
<ul>
  <li><b>Geographic Feature Derivation:</b> Engineered a dynamic <code>vehicle_age</code> vector by cross-referencing and parsing age metrics from UK structural string <code>reg_code</code> variables.</li>
  <li><b>Leakage-Free Imputation:</b> Imputed continuous elements via the feature <b>median</b> and sparse categorical values using the column <b>mode</b> (<code>SimpleImputer</code>). Implemented explicit binary missingness indicators to capture latent data patterns.</li>
  <li><b>Outlier Regularisation:</b> Applied localised extreme-value clipping bounding features strictly between the <b>1st and 99th percentiles</b>.</li>
</ul>

<h3>2. High-Cardinality Encoding Strategy</h3>
<ul>
  <li><b>Smoothed Target Encoding:</b> Deployed continuous target encoding (<code>smooth=10.0</code>) onto massive discrete feature spaces (<code>standard_make</code>, <code>standard_model</code>) to naturally compress categorical dimensions without inflating framework sparsity.</li>
  <li><b>Rare-Category Consolidation:</b> Grouped low-frequency categorical occurrences (&lt; 1% frequency threshold) into a uniform 'Other' bucket prior to standard One-Hot Encoding.</li>
  <li><b>Standardisation:</b> Passed continuous feature matrices through a <code>StandardScaler</code> pipeline.</li>
</ul>

<h3>3. Automated Feature Selection & Dimensionality Reduction</h3>
<ul>
  <li><b>Recursive Feature Elimination (RFECV):</b> Utilised backward elimination cross-validation driven by a linear base estimator to shrink the feature universe to the 28 most predictive channels.</li>
  <li><b>Linear Dimensionality Reduction (PCA):</b> Constructed an optimal Principal Component Analysis pipeline, mapping a Scree Plot to isolate the components required to retain <b>&gt; 90% of global variance</b>.</li>
  <li><b>Non-Linear Manifold Projection:</b> Visualised and contrasted non-linear space compression using <b>Isomap</b> (preserving geodesic distances) alongside <b>t-SNE</b> (preserving local neighbourhoods) to map distinct pricing regions.</li>
</ul>

<h3>4. Advanced Non-Linear Modelling</h3>
<ul>
  <li><b>Bayesian Polynomial Curve Fitting:</b> Isolated the <code>vehicle_age</code> feature to generate 3rd-degree polynomial combinations, fitting a <code>BayesianRidge</code> regressor to track compounding depreciation curves alongside explicit <b>95% credible intervals</b>.</li>
  <li><b>Tuned Ensembles:</b> Designed, parameterised, and cross-validated high-capacity tree ensembles using <code>GridSearchCV</code>, explicitly optimising <code>RandomForestRegressor</code> and <code>GradientBoostingRegressor</code> instances.</li>
  <li><b>Meta-Ensembling:</b> Combined optimised base models into higher-tier architectures using a <b>Voting Regressor</b> (averaging predictions) and a <b>Stacking Regressor</b> utilising a <code>RidgeCV</code> meta-learner.</li>
</ul>

<h3>5. Explainable AI (XAI) & Market Clustering</h3>
<ul>
  <li><b>SHAP Model Introspection:</b> Deployed a <code>TreeExplainer</code> shell over our top ensemble to extract global summary dependencies and individual local waterfall attribution maps.</li>
  <li><b>Multi-Algorithm Segmentation Space:</b> Grouped vehicles by their underlying SHAP profile distributions using <b>K-Means Clustering</b> (optimised via the Elbow Method at <b>k=4</b>), structural <b>Hierarchical Ward-linkage dendrograms</b>, and density-based <b>DBSCAN</b> sweeps to map out hidden consumer segments.</li>
  <li><b>Feature Enrichment:</b> Injected these latent market segment labels (<code>shap_cluster_group</code>) directly back into the primary pipeline as an engineered feature for final model retraining.</li>
</ul>

<hr />

<h2>📈 Experimental Performance Results</h2>
<p>
  Every model configuration was cross-validated and validated against unseen validation splits, with target outputs transformed back using <code>np.expm1()</code> to evaluate final metrics in true currency (£).
</p>

<h3>Baseline & Tree Ensembles</h3>
<table width="100%">
  <thead>
    <tr bgcolor="#161b22">
      <th align="left">Model Configuration</th>
      <th align="center">Validation R² Score</th>
      <th align="center">Validation MAE (£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest Regressor (Tuned)</td>
      <td align="center">0.9209</td>
      <td align="center">£3,151.79</td>
    </tr>
    <tr bgcolor="#1f242c">
      <td>Gradient Boosting Regressor (Tuned Base)</td>
      <td align="center">0.9252</td>
      <td align="center">£3,128.81</td>
    </tr>
  </tbody>
</table>

<h3>Meta-Ensembles & Feature-Enriched Pipelines</h3>
<table width="100%">
  <thead>
    <tr bgcolor="#161b22">
      <th align="left">Model Configuration</th>
      <th align="center">Validation R² Score</th>
      <th align="center">Validation MAE (£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Stacking Meta-Regressor (<code>RidgeCV</code>)</td>
      <td align="center">0.9268</td>
      <td align="center">£3,096.93</td>
    </tr>
    <tr bgcolor="#1f242c">
      <td><b>Voting Meta-Regressor (Best Ensemble)</b></td>
      <td align="center"><b>0.9274</b></td>
      <td align="center"><b>£3,061.99</b></td>
    </tr>
  </tbody>
</table>

<p>
  <i><b>Key Takeaway:</b> The Meta-Ensemble strategies successfully bypassed individual algorithm limitations, while injecting latent market clusters allowed the final architectures to better account for non-linear interactions across distinct automotive sub-markets.</i>
</p>

<hr />

<h2>🚀 How to Run the Project</h2>
<ol>
  <li>
    <b>Clone the repository:</b>
    <pre><code>git clone https://github.com/Ashwashhere/AdvancedMachineLearning_CarValuation.git
cd AdvancedMachineLearning_CarValuation</code></pre>
  </li>
  <li>
    <b>Execute the workspace pipeline:</b>
    <p>Run the compiled script profile directly via your terminal shell environment:</p>
    <pre><code>python3 AMLThumbnail.py</code></pre>
  </li>
</ol>
