# Flow ROP · KRLS Explorer

Streamlit predictor for DBU-catalyzed lactide ring-opening polymerization. Fits separate RBF kernel ridge regression surrogates for the initiator concentrations provided in `ALL.csv` (5, 10, 20 mM). Predictions include experimental Mn in **kDa**, dispersity and fractional conversion. Rates are calculated using a reactor volume of **0.8345 mL**, monomer concentration **1 M**, and molecular weight **144.126 g/mol**, matching the spreadsheet convention. The rate per DBU is divided by the **numeric DBU concentration in mM** as in the source CSV.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Input DBU and residence time inside the observed rectangular bounds for the chosen initiator concentration.
- Predict Mn, Đ, conversion, mass production and rate divided by DBU concentration.
- Inverse-design optimizer: find a specified production rate with maximum conversion, or a specified conversion with maximum production rate; flag infeasible targets.
- Interactive choice among five predicted contour maps, with experimental locations plotted.

## Method and limitations
This uses **batch RBF kernel ridge regression**, a regularized kernel least-squares surrogate closely related to KRLS; it may not reproduce the exact KRLS software or smoothing choices previously used. Training predictors are experimental `Mnexp`, `dispersity` and `conversion`. Inverse optimization is constrained to the observed rectangular coordinate ranges, not necessarily the convex hull of measured experiments. Other conditions are assumed equal to the recorded experiments. This is a research explorer, not a guaranteed synthesis outcome.
