# Response to reviewers

We appreciate the time the reviewers took to offer suggestions for improvement, with their inclusion in the manuscript materially improving both the underlying science and its readability.

General response to editor and reviewers discussing the main thrusts of the edits.

In addition to these general changes, we also include specific responses to each of the comments provided by the reviewers, along with the locations in the updated manuscript of the implemented changes.

## Reviewer 1

> The amount of data used is extremely limited compared to the amount collected and available from OIB. It would be useful to significantly expand the amount of data used in this analysis and to discuss cross-region variability of the results.

We agree with the reviewer that it would be useful to expand the analysis to a wider region, with a discussion of cross-region variability.
We, however, feel such an investigation is outside the intended scope of the current study.
We have limited our analysis to regions where we have higher confidence in our depth-density modeling results (due to adequate *in-situ* density record coverage) and where repeat OIB flights with a number of years of separation between collection time exist.
Although research is ongoing to extend the density model used to wider regions of Antarctica, it is not possible to complete and implement this work in the short time frame required for revisions.
We have added additional text to the manuscript more explicitly stating the limits of this study, with hope to further expand it in the future.

> There should be some discussion and analysis of the effect of layer-echo signal to noise ratio in the precision of the layer measurements and the resulting accumulation estimates.

We were uncertain of the precise issue raised by this comment.
We understood it to mean...
The original paper outlining the PAIPR method discusses these concerns in greater detail.
*Discussion of uncertainty incorporation in PAIPR.*
We have included additional discussion in the manuscript to better highlight these points, and hope such discussion will satisfactorily resolve this concern.

## Reviewer 2

> The 2011 and 2016 transects were actually flown with slightly different radar parameters. 2011 had a bandwidth of 4 GHz (2-6 GHz sweep) and 2016 has a bandwidth of 6 GHz (2-8 GHz sweep). (If you’re using the CReSIS MATLAB files from their ftp site, you can check these parameters in the param_get_heights.radar.wfs variable that comes with the data.) This isn’t a huge difference in range resolution (~4 cm vs ~2.7 cm in snow), but I would expect some impact on the layer tracking results.This seems worth mentioning, both as a potential source of both the error and bias and because your results now actually suggest that the accumulation estimates are fairly robust to these kinds of small variations in the bandwidth/range resolution.

Blah blah blah.

> How did you choose the $S_Q = 0.50$ cutoff and the 10% quality cutoff for radargrams? Are those values you would expect to be reasonable for any data sets, or is there a logic to their choice that would be a useful for a reader to understand if they were interested in using these methods themself?

Blah blah blah.

> It was not totally clear to me why there is uncertainty/variability in the value of the logistic function parameters and how the ranges are chosen for the Monte Carlo implementation. If this can be clarified in a sentence or two, that  would be  nice  for readers  like  myself who are  not  fully familiar with the PAIPR method.

Blah blah blah.

> Reference of the in-situ accumulation estimates that are used for the bias distributions? It would be nice to have a quick sentence explaining where that data comes from and a pointerto any papers on it.

We believe poor wording on our part led to some confusion regarding this point, as more than one reviewer mentioned it.
We do not use other methods to compare to radar results (e.g. ice cores, climate model outputs, etc.) but instead focus exclusively on 

> Is the mean in-situ accumulation rate one value over the entire study area?

Blah blah blah.

> I find the spatial correlation in bias shown in Fig. 3 very interesting and I think it’s quite an important result that suggests that mean error from sparse crossovers is not a robust way to quantify uncertainty in these types of radar measurements. I am curious whether you have looked at the spatial correlation of this bias pattern with either radar system parameters or physical conditions on the ice sheet beyond the accumulation  rate.  For  example,  I  could  imagine  that  variations  in  signal-to-noise/signal-to-clutter ratio/image quality rating, aircraft roll, or surface slope/surface roughness/topographic variability could play a role in the stability of the estimates. If so, these could be useful insights for future users of these methods when making some initial assessment of their study region and the available data.

We appreciate this suggest and agree some discussion of these correlations would be beneficial.
Although a full investigation of these relationships is beyond the scope of this short paper, we have included additional language briefly discussing this and our key findings regarding these connections.

> Consider briefly mentioning surface velocities for this region somewhere in the paper since horizontal advection might contribute to observed uncertainty in some of the faster flowing regions.  Although it doesn’t look off-hand like there is a correlation between ice velocity and bias in the study area, which is great.

Blah blah blah.

> Do you have any insights from this work on the length scale over which measurements can be considered repeatable? I’m assuming from the high RMSE that variability at the trace-by-trace level is quite high, but that averaged over some distance the agreement between seasons improved. If so that, would also be a useful note to add to the discussion/conclusions.

Blah blah blah.

> Fig. 1 - the color gradient on the flight lines is somewhat difficult to see. I would suggest making the flight line thicker.

We thickened the lines to better show the accumulation variability in this figure.

> Fig. 3 – consider shrinking the mean bias colorbar to the height of panel b, since it only applies to that panel.

We have shrunk the colorbar to the vertical limits of the lower panel.

> Fig. S2 –I assume  that  the dashed lines show some form of uncertainty bounds? Is this $\pm1 \sigma$, 95% confidence intervals, or something else? Please clarify in the caption.

We have addressed with oversight and explicit state in the figure caption that these lines represent $\pm1 \sigma$ errors.

## Reviewer 3

> The error analysis could be improved significantly by comparing the differences between each method with the error one would expect from repeat manual layer tracing.

Blah blah blah.

> Additionally, a comparison of accumulation results to regional climate model output would be a better way to determine “bias,” since those models are used throughout the literature and in policy recommendations.

Although we agree that comparisons to climate model outputs are valuable comparisons, they do not fit within the main focus and direction for this article.
This paper focuses on directly comparing radar results from 2011 and 2016, without attempting to assess whether the results are accurately representing annual accumulation rates (other studies focus more specifically on that question---Medley, Lewis, Keeler, etc.).
We instead investigate whether estimates derived from radar remain consistent and repeatable over a 5-year period, with the main goal assessing the suitability of comparing/combining accumulation estimates from differing radar collection times.
Additionally, the more coarse spatial footprint associated with climate modeling outputs would results in ambiguity in whether the differences with radar-derived estimates result from simple spatial variability in accumulation rates or from some limitation/inaccuracy in the radar-derived results.

> “Reconstructed from airborne radar imagery collected 5 years apart” could be “collected in 2011 and 2016” to reduce confusion.

We have incorporated the suggested wording (Line ).

> “both collection times” – refers to 2011 and 2016, but is not very clear in the abstract.

We have incorporated the suggested wording (Line ).

> Hyphen between “ice” and “penetrating” here, but not earlier in the abstract

We corrected this typo (Line ).

> Last sentence is a bit of a run-on. I suggest removing “in many cases” and splitting sentences after “by several years”

We removed the extraneous phrase and split into two sentences.

> Should “dominant control on sea level rise” be “dominant contributor to?”

We replaced "control on" with "contributor to" (Line ).

> SMB is not just an “important component of ice sheets,” but is now the single dominant component of ice sheet mass loss (van den Broeke et al., 2009, 2016).

Blah blah blah

> Change “this encompasses freshly-fallen...” to “this balance” or “this sum” or similar.

We have incorporated the suggested wording (Line ).

> Could cite a few ground-based ice penetrating radar papers to give validity to the airborne method and show how much more spatial coverage IceBridge can achieve in one season

Blah Blah blah.

> Give an end date for “NASA’s Operation IceBridge (OIB) campaign (commenced in 2009)”

Included the 2019 end of OIB.

> The Snow Radar does not “image” the subsurface, rather it collects geophysical data one can use to calculate SMB, or something similar

We have replaced "permit imaging" to "permit discernment" to better adhere to the precise definition in regards to radiostratigraphy.

> If the authors bring up “climate modeling” here then the reader would expect some comparison of Snow Radar accumulation rates to regional climate model output. See Koenig et al (2016 Figure 10 for example). RCM rates should be the accumulation rates you compare against, since they are the widely accepted “correct” values

Blah blah blah

> “...can change on shorter time scales” could be sub-annual time scales? Be more specific

Blah blah blah.

> Both [12 - Figure 8] and [13 – Figure 4] use OIB flights from different years to show “temporally stable” measurements at the cross-over points.

We include additional wording to better clarify the distinction between our work on prior studies.
Blah blah blah...

These crossover studies show internal consistency with increasing depth (temporal stability in the sense of repeatable reconstructions across the full time series of the record) but do not assess the effect of temporal stability in regards to time passing between data collection (i.e. they do not distinguish crossovers for the same flight, flights from the same year, flights spaced multiple years apart in time, etc.).
Koenig et al. (2016) includes additional snow deposition between flights as a potential source of error, but does not investigate the magnitude of this effect on estimate error or how the error changes as time between flights increases.
Lewis et al. (2017) similarly make no distinction in their crossover analysis for the timing of radar collection but are instead focused only on the spatial overlap between flights.
Additionally, Lewis et al. (2017) use flights only 2013--2014 OIB flights, limiting the potential effects of temporal instability that is the focus of our manuscript.

> “Most of these studies also focus on multi-decadal mean accumulation rates...” – can’t completely ignore Koenig’s snow radar work, which focused on single year accumulation rates from Snow Radar in Greenland. This manuscript is analyzing 1990-2010 accumulation rates, which is “multi-decadal mean accumulation rates.” This sentence seems to lead the reader to a different conclusion than where this manuscript is heading.

We have edited this section to increase clarity, and have explicitly distinguished Koenig et al. (2016, maybe others?) as also focusing on single year accumulation rates.

> “regardless of collection year” is a bit of a misnomer. You can’t trace annually resolved layers from 100+ years ago with the Snow Radar. So maybe “in recent decades” or similar

Blah

> “within 500 m of one another” – how much variability do you expect within those 500 m? Authors should reference some repeat ice core measurements, RCMs, or the Greenland Bamboo Forest to state how much variability/uncertainty in accumulation exists between these “overlapping” flights, which could then make your bias calculations look even more accurate.

Blah

> 