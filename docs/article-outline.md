# Recent trends and patterns of surface mass balance variability in central West Antarctica

## Outline

### Introduction

Here I will discuss the motivation and background of the project.
Some of the primary motivations include:

- SMB varies greatly across the ice sheet, but constrained by few *in-situ* data points
- Radar is a valulabe tool, but requires automated methods to best use large datasets
- Such methods require careful vetting to ensure consistency of results prior to use in interpretations

Summary paragraph of what we did (with concluding summary of why/significance of study).

### Methods

Primary method is discussed in detail in Keeler et. al, 2020.
See Supplement 1 for changes and updates to methodology since publication.
The most relevent changes include:

- Change from layer segment generation to summed streamlines
- An automated QC routine based on consistency of results in the radon transforms step
- Use of improved depth-density modeling (see White et. al, 2020)

Description of bootstrapping for trend analysis.
Description of methods for spectral analysis.

### Results

Discuss/show plots of repeatability results.
Discuss issues observed and noteworthy observations in repeatability.
Summarize to what extent our results are repeatable both intra-flightline and inter-flightline (hopefully within the uncertainty of annual estimates).
Will need to dicuss repeatability for both annual results and trend results.

Discuss patterns in the accumulation results:

- Mean accumulation patterns match observations seen in cores (and reanalysis?)
  - Higher accumulations near the coast, with gradation to lower accumulations in the interior
  - Gradient with higher accumulation to the east and lower to the west
- Dipole in accumulation trends, with more drastic losses to the west and more muted losses/gains to the east
  - How significant are these trends? Need to be careful here with how far we infer from these results.
- What do the trends/patterns look like if we just use cores? Do they match reasonably well, given the low sample size of the cores?

Discuss frequency modes from spectral analysis:

- Dominated by lower frequency (multi-year to decadal signals) modes
- Summary of estimates (with uncertainties hopefully)
- Highlight any regions of differing but spatially coherent modes
- How well do these results line up with the limited core data?
- Should we show how well the results compare with reanalysis?

## TODO

### Core PAIPR functionality

- Optimization of QC cutoff values
  - Not entirely sure the best way to do this
- Incorporation of logistic regression parameters into the Monte Carlo simulations
  - Likely need to increase MC simulations to at least 1,000
  - Also determine whether regression parameters have spatial component and model accordingly?
- Improved surface assignment (right now performed using constant offset)
- Re-optimization of logistic parameters based on more manual picks (from repeatability efforts)
- Address multi-modal nature of accumulation distributions (right now I'm thinking about performing some sort of test for multi-modal behavior, and if results are sufficiently mutlti-modal, focusing only on the major mode)
  - I could also keep the minor mode if it is at least X% of the major mode)

### Investigations of repeatability

I need to determine how repeatable and reproduceable the results are between different flightlines.
Steps to address this include:

- More manual picks of repeat flightlines, to determine if discrepencies lie with PAIPR or radar generally (may also result from density model)
- Repeat tests for other flightlines (max of 1 year apart) to determine if the divergence is time-dependent
- Aggregate/average results over regions (1 km^2^?) to increase number of comparison points?
- Test removal of banked flight data (talk to Clem about addressing this) to see if this improves results
- Test handful (3-5) of flightline segments for repeatability between data_smooth and data_stack results

### Spectral analysis

- Ensure low frequency modes are real and not some processing artefact
- Determine extent of memory (red noise) in the system by looking at lagged autocorrelation
  - e-folding lag for autocorrelation drop-off?
- Determine how we can incorporate the uncertainty in our data to these results (bootstrapping again?)
  - How do I determine the significance of the observed frequency modes?

## List of issues/concerns

This is a running list of any miscellaneous issues or concerns not already covered in the TODO list above.

- Because the errors increase with depth, is there a risk of inducing an artificial negative trend because the range of our distribution grows larger at earlier times (and the estimates are bounded at 0)?
  - Test by seeing if large negative trends are only for regions with large uncertainties at the earliest years
