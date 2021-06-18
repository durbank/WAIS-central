---
geometry: margin=2.5cm
---

# Response to reviewers

We appreciate the time the reviewers took to offer suggestions for improvement, with their inclusion in the manuscript materially improving both the underlying science and its readability.
We include specific responses to each of the comments provided by the reviewers, along with the locations in the updated manuscript of the implemented changes.
Although we feel all the suggested edits were valuable, we believe the sum total of all these edits exceeds what can adequately be covered in such a short and targeted study.
We have therefore incorporated all the suggestions we reasonably can while judiciously deferring less applicable suggestions to future endeavors.
For each instance where we elected not to incorporate the suggested edit, we have detailed our reasoning and thoughts as to why such suggestions, while objectively meritorious, are best pursued in later work.

The edits incorporated into the manuscript also necessitated the moving of the discussion of updates to the PAIPR methodology and the figure showing representative echograms to the supplement.
We do not, however, feel these changes materially impact the readability of the manuscript or cause undue jumps in logic when trying to follow the methods, results, and implications of the research.
We have made edits throughout the manuscript to ensure no loss in continuity due to these adjustments.

We include two copies of the manuscript in this submission.
The first is a direct update from the initial submission in the.
The second highlights the specific changes we have made since the original submission with additions in blue and deletions in red.
Line numbers in the following comments reference the location of edits in the second, color-highlighted document.

## Responses to Reviewer 1

> The amount of data used is extremely limited compared to the amount collected and available from OIB. It would be useful to significantly expand the amount of data used in this analysis and to discuss cross-region variability of the results.

*We agree with the reviewer that it would be useful to expand the analysis to a wider region, with a discussion of cross-region variability.
We, however, feel such an investigation is outside the intended scope of the current study.
We have limited our analysis to regions where we have higher confidence in our depth-density modeling results (due to adequate *in-situ* density record coverage) and where repeat OIB flights with a number of years of separation between collection time exist.
Although research is ongoing to extend the density model used to wider regions of Antarctica, it is not possible to complete and implement this work in the short time frame required for revisions.
We nonetheless believe the data included in the study is sufficiently spatially extensive to be useful and informative to the wider scientific community.*

> There should be some discussion and analysis of the effect of layer-echo signal to noise ratio in the precision of the layer measurements and the resulting accumulation estimates.

*SNR of echo layers are relevant to this research, so we are glad the reviewer raised this point.
We have added a discussion of the impact of radar attenuation on SNR and its effect on layer selection (Lines ).
We also briefly discuss how these impacts are limited in our study (Lines ).*

## Responses to Reviewer 2

> The 2011 and 2016 transects were actually flown with slightly different radar parameters. 2011 had a bandwidth of 4 GHz (2-6 GHz sweep) and 2016 has a bandwidth of 6 GHz (2-8 GHz sweep). (If you’re using the CReSIS MATLAB files from their ftp site, you can check these parameters in the param_get_heights.radar.wfs variable that comes with the data.) This isn’t a huge difference in range resolution (~4 cm vs ~2.7 cm in snow), but I would expect some impact on the layer tracking results. This seems worth mentioning, both as a potential source of both the error and bias and because your results now actually suggest that the accumulation estimates are fairly robust to these kinds of small variations in the bandwidth/range resolution.

*We appreciate the reviewer bringing up this point and its significance to our results.
We edited the manuscript to explicitly note this difference in bandwidth between the two flights and its impact on range resolution (Lines ).
We also now mention this difference when discussing potential sources of error in our results and indicate that our results suggest such variations have minimal impact on the reproducibility of SMB estimates from radar (Lines ).*

> How did you choose the $S_Q = 0.50$ cutoff and the 10% quality cutoff for radargrams? Are those values you would expect to be reasonable for any data sets, or is there a logic to their choice that would be a useful for a reader to understand if they were interested in using these methods themself?

*This comment raises a good point regarding the selection of these thresholds.
We have added additional discussion on the $S_Q$ parameter to better give intuition to its physical meaning (Lines ).
We further include a brief discussion regarding the choice of parameter values, the rationale behind that choice, we indicate that a small amount of the overall dataset (~3%) were removed through this, and present some evidence supporting our choice of parameter values (Lines ).*

> It was not totally clear to me why there is uncertainty/variability in the value of the logistic function parameters and how the ranges are chosen for the Monte Carlo implementation. If this can be clarified in a sentence or two, that  would be  nice  for readers  like  myself who are  not  fully familiar with the PAIPR method.

*We have added clarifying language to address this confusion (Lines ).
Essentially we optimize the logistic function parameters using manual layer traces, which we repeat across all traces of multiple echograms.
This leads to estimated parameter values from each trace, and a distribution of ~15,000 estimates across the full training dataset.
We use this distribution to inform our expected value and variability of logistic parameters when performing our Monte Carlo simulations.*

> Reference of the in-situ accumulation estimates that are used for the bias distributions? It would be nice to have a quick sentence explaining where that data comes from and a pointer to any papers on it.

*We believe poor wording on our part led to some confusion regarding this point, as more than one reviewer mentioned it.
We do not use other methods to compare to radar results (e.g. ice cores, climate model outputs, etc.) but instead focus exclusively on the biases and errors between radar results from different methods and flights.
The target of this research is not to determine whether radar-derived accumulation is accurate compared to other independent methods, but rather whether radar-derived estimates are self-consistent over time.
We therefore do not perform direct comparisons between radar results and independent methods (e.g. firn cores, RCM outputs, etc.).
We have modified language throughout the manuscript to make this point more clear.
We have specifically removed all references to *in-situ data* throughout the manuscript to avoid additional confusion.*

> Is the mean in-situ accumulation rate one value over the entire study area?

*This mean value refers to the location-specific mean accumulation across all four methods for a given year.
It therefore is not a single consistent value for the whole region but instead varies by location.
We elected to use a spatially-varying mean to scale our biases due to the large gradient in accumulation rates across the region (varying by a factor of ~3).
We have removed the term *in-situ* and replaced it with wording that more clearly articulates this definition (Lines ).*

> I find the spatial correlation in bias shown in Fig. 3 very interesting and I think it’s quite an important result that suggests that mean error from sparse crossovers is not a robust way to quantify uncertainty in these types of radar measurements. I am curious whether you have looked at the spatial correlation of this bias pattern with either radar system parameters or physical conditions on the ice sheet beyond the accumulation  rate.  For  example,  I  could  imagine  that  variations  in  signal-to-noise/signal-to-clutter ratio/image quality rating, aircraft roll, or surface slope/surface roughness/topographic variability could play a role in the stability of the estimates. If so, these could be useful insights for future users of these methods when making some initial assessment of their study region and the available data.

*We appreciate this suggestion and agree some discussion of these correlations would be beneficial.
Although a full investigation of these relationships is beyond the scope of this short paper, we have included additional language briefly discussing this and our key findings regarding these connections (Lines ).
We find no obvious correlations between the strength of the bias and surface slope, surface aspect, plane altitude above the ice sheet surface, aircraft pitch, or aircraft roll.
We see a positive correlation between surface elevation and the bias magnitude but believe this is an expression of lower accumulation rates at higher elevations that increases the error associated with SMB estimates (and therefore increases the potential for larger biases).*

> Consider briefly mentioning surface velocities for this region somewhere in the paper since horizontal advection might contribute to observed uncertainty in some of the faster flowing regions.  Although it doesn’t look off-hand like there is a correlation between ice velocity and bias in the study area, which is great.

*We have added language discussing the range in ice velocities for the region and our rationale for considering horizontal advection to be negligible for most of our data in this study (Lines ).*

> **Do you have any insights from this work on the length scale over which measurements can be considered repeatable? I’m assuming from the high RMSE that variability at the trace-by-trace level is quite high, but that averaged over some distance the agreement between seasons improved. If so that, would also be a useful note to add to the discussion/conclusions.**

*I don't really have the space to go into this at all, so any suggestions on how best to respond to this comment?*

> Fig. 1 - the color gradient on the flight lines is somewhat difficult to see. I would suggest making the flight line thicker.

*We thickened the lines to better show the accumulation variability in this figure.
We additionally made a number of other changes to this figure to improve the amount and form of the information conveyed.
We have increased the size of all fonts to aid in visibility.
We have also modified the color of the thicker line (denoting overlap) to better contrast with the accumulation color mappings.
We have further incorporated additional modifications to the figure (addition of hillshade and contour lines) to provide better contextual information of the region of study.*

> Fig. 3 – consider shrinking the mean bias colorbar to the height of panel b, since it only applies to that panel.

*We have shrunk the colorbar to the vertical limits of the lower panel.
We have additionally increased the size of fonts to aid in readability.
We have further incorporated additional modifications to the figure (addition of hillshade and contour lines) to provide better contextual information of the region of study.*

> Fig. S2 –I assume  that  the dashed lines show some form of uncertainty bounds? Is this $\pm1 \sigma$, 95% confidence intervals, or something else? Please clarify in the caption.

*We have addressed with oversight and explicitly state in the figure caption that these lines represent $\pm1 \sigma$ errors.*

## Responses to Reviewer 3

> **The error analysis could be improved significantly by comparing the differences between each method with the error one would expect from repeat manual layer tracing.**

*I'm still not really sure how to respond to this. It is not something that will be addressed in the revised manuscript so any recommendations on the best way to refute the point?*

> Additionally, a comparison of accumulation results to regional climate model output would be a better way to determine “bias,” since those models are used throughout the literature and in policy recommendations.

*Although we agree that comparisons to climate model outputs are valuable comparisons, they do not fit within the main focus and direction for this article.
This paper focuses on directly comparing radar results from 2011 and 2016, without attempting to assess whether the results are accurately representing annual accumulation rates (other studies focus more specifically on that question---e.g. Medley et al. (2013), Lewis et al. (2017), Keeler et al. (2020), etc.
The target of this research is not to determine whether radar-derived accumulation is accurate compared to other independent methods, but rather whether radar-derived estimates are self-consistent over time.
We therefore do not perform direct comparisons between radar results and independent methods (e.g. firn cores, RCM outputs, etc.).
We instead investigate whether estimates derived from radar remain consistent and repeatable over a 5-year period, with the main goal assessing the suitability of comparing/combining accumulation estimates from differing radar collection times.
Additionally, the more coarse spatial footprint associated with climate modeling outputs (10's of km vs ~200 m) specifically would result in ambiguity in whether the differences with radar-derived estimates result from simple spatial variability in accumulation rates or from some limitation/inaccuracy in the radar-derived results.*

> “Reconstructed from airborne radar imagery collected 5 years apart” could be “collected in 2011 and 2016” to reduce confusion.

*We have incorporated the suggested wording (Line ).*

> “both collection times” – refers to 2011 and 2016, but is not very clear in the abstract.

*We have incorporated the suggested wording (Line ).*

> Hyphen between “ice” and “penetrating” here, but not earlier in the abstract

*We corrected this typo (Line ).*

> Last sentence is a bit of a run-on. I suggest removing “in many cases” and splitting sentences after “by several years”

*We removed the extraneous phrase and split the sentence into two sentences.*

> Should “dominant control on sea level rise” be “dominant contributor to?”

*We replaced "control on" with "contributor to" (Line ).*

> SMB is not just an “important component of ice sheets,” but is now the single dominant component of ice sheet mass loss (van den Broeke et al., 2009, 2016).

*We have added additional wording better highlighting the relevance of SMB, with particular reference to the work of van den Broeke et al. (Lines )*

> Change “this encompasses freshly-fallen...” to “this balance” or “this sum” or similar.

*We have incorporated the suggested wording (Line ).*

> Could cite a few ground-based ice penetrating radar papers to give validity to the airborne method and show how much more spatial coverage IceBridge can achieve in one season

*We have added additional citations of ground-based radar studies to better highlight the increased spatial coverage of Operation IceBridge.*

> Give an end date for “NASA’s Operation IceBridge (OIB) campaign (commenced in 2009)”

*We added the OIB end year of 2019.*

> The Snow Radar does not “image” the subsurface, rather it collects geophysical data one can use to calculate SMB, or something similar

*We have replaced "permit imaging" to "permit discernment" to better adhere to the precise definition in regards to radiostratigraphy.*

> If the authors bring up “climate modeling” here then the reader would expect some comparison of Snow Radar accumulation rates to regional climate model output. See Koenig et al (2016 Figure 10 for example). RCM rates should be the accumulation rates you compare against, since they are the widely accepted “correct” values

*The reference to climate modeling here is merely summarizing the findings of the cited study used as an example of rapidly changing surface conditions.
Nonetheless, we have removed this example as to avoid inadvertently leading the reader to conclusions of RCM comparisons (Lines ).*

> “...can change on shorter time scales” could be sub-annual time scales? Be more specific

*We have explicitly stated we are referring to time scales of a few years.*

> Both [12 - Figure 8] and [13 – Figure 4] use OIB flights from different years to show “temporally stable” measurements at the cross-over points.

*We include additional wording to better clarify the distinction between our work and prior studies (Lines ).
We specifically highlight the differences between the crossover analyses performed in Lewis et al. (2017) and Koenig et al. (2016), and why our approach better targets the specific question of repeatability with disparate collection times specifically.*

> “Most of these studies also focus on multi-decadal mean accumulation rates...” – can’t completely ignore Koenig’s snow radar work, which focused on single year accumulation rates from Snow Radar in Greenland. This manuscript is analyzing 1990-2010 accumulation rates, which is “multi-decadal mean accumulation rates.” This sentence seems to lead the reader to a different conclusion than where this manuscript is heading.

*We have edited this section to increase clarity, and have explicitly distinguished Koenig et al. (2016) as also focusing on single year accumulation rates while highlighting again the difference in and significance of our methodological focus (Lines ).*

> “regardless of collection year” is a bit of a misnomer. You can’t trace annually resolved layers from 100+ years ago with the Snow Radar. So maybe “in recent decades” or similar

*We have edited the sentence to reflect that we are specifically interested in understanding repeatability of radar-derived SMB collected over a period of years.*

> “within 500 m of one another” – how much variability do you expect within those 500 m? Authors should reference some repeat ice core measurements, RCMs, or the Greenland Bamboo Forest to state how much variability/uncertainty in accumulation exists between these “overlapping” flights, which could then make your bias calculations look even more accurate.

*This is an important point that we appreciate the reviewer raising.
We have added two citations giving context to the magnitude of small-scale spatial variability in Antarctica (Lines ) and also again make reference to this variability in our conclusions (Lines ).*

> “additional details on this unit” do you mean “radar” or “radar unit”? A bit confusing as worded

*We have changed this to "Snow radar" to more clearly convey the unit in question.*

> “except we apply the model proposed in [20]...” Could you fit one sentence in this manuscript briefly describing that method? The depth-density relationship is critical for the rest of this analysis, so I feel it’s important to include

*We added additional sentences to describe this method and the specific advantages/rationale for using it (Lines ).*

> What does “expert manual tracing” entail? One author tracing the lines once? Or multiple authors? Or multiple rounds of tracing? Etc.

*We have added context to explicitly show that these manual estimates are based on a single iteration of layer selection (Lines ).*

> “covering a range of accumulation rates and subsurface” – please provide the range here so it’s not just in the Figure 1 caption

*We have added the precise range in accumulation rates to this section of the text (Lines ).*

> “The error for the PAIPR method incorporates” implies that the “expert manual tracing” has zero error, when there certainly is some uncertainty

*We have added clarifying language discussing that the manual method indeed entails additional uncertainty, but that we do not attempt to quantify it due to the subjective nature of the method (Lines ).
Even using repeat iterations of manual tracing (either by the same individual or multiple individuals) would reflect psychological and physiological influences as much or more than the true error associated with manual tracing methods.
We instead use this to highlight one of the key advantages of an automated method like PAIPR in that it allows for more rigorous quantification of this uncertainty (Lines ).*

> “and 1 (most suspect)” is not a proper scientific term

*We have removed the offending phrase (Line ).
We have also expanded our discussion of the image quality scoring metric to clarify what these values represent (Lines ).*

> IRH is defined after Equation 1, but also used in the paragraph before

*We now properly define this acronym prior to its first use (Line ).*

> What values of S_Q do you see throughout your data? One sentence describing the quality scores would be helpful

*We have expanded our discussion of the image quality scoring metric to clarify what these values represent, and the choice parameter values $S_Q$ and the cutoff percentage threshold (Lines ).*

> Can you give a few examples of “regions where IRHs are discontinuous?” How much of Antarctica (or the region you studied) would fall into that category? Does SEAT2010-4 have a quality control flag?

*We have added a sentence stating such discontinuities are more common in coastal regions and areas that experience melt/refreeze and are therefore less impactful on our specific study (Lines ).
We added a note specifying that 3.1% of the original data results were removed by the automated quality control flags (Lines ).
We added a note stating that issues in the Site B vicinity were not associated with QC flagging but instead this region shows clear, well-delineated layers in both 2011 and 2016 results that do not agree with each other (Lines ).*

> “depths below the surface represented in the results will differ” – can you quantify how much deeper the IRHs are using the 2016 data? Just a range or mean and standard deviation would do

*We have added the mean difference in depth between 2016 and 2011 data for both the 2010 and 1990 horizons (Lines ).*

> The “IV. Results” is hidden above Figure 3, I suggest that the typesetting team make sure this section heading is visible

*We leave this to the typesetting team to resolve in the final version.*

> What are you considering “correct” when computing “the differences, biases, and errors?” Could be a good place to compare with RCMs or field measurements (if they exist).

*We believe poor wording on our part led to some confusion regarding this point, as more than one reviewer mentioned it.
We do not use other methods to compare to radar results (e.g. ice cores, climate model outputs, etc.) but instead focus exclusively on the biases and errors between radar results from different methods and flights.
The target of this research is not to determine whether radar-derived accumulation is accurate compared to other independent methods, but rather whether radar-derived estimates are self-consistent over time.
We therefore do not perform direct comparisons between radar results and independent methods (e.g. firn cores, RCM outputs, etc.).
We instead use the mean accumulation value at a given location and year to determine percent differences between different methods and flight data.
We have modified the wording referencing this throughout the manuscript to hopefully make this distinction more clear.*

> “2016 estimates on average are 2.98...” – sentence starts with a number. Estimates of what? 

*We have fixed the typos to ensure what is discussed is clearly evident (Line ).*

> How important is this 3% differences compared with tracing IRHs twice for the same flight? Or for two flights within 500 m of each other?

*We added context for this difference compared to small-scale spatial variability (Lines ), reinforcing the conclusion that the measured bias between flight results is relatively small.*

> “when expressed as % bias relative to mean in-situ  annual accumulation” replace with “percent.” What are you using as the “correct” accumulation?

*We have fixed the typo (Line ) and added language to clarify that this % bias is relative to the average accumulation across all methods for a given location and year (Lines ).*

> “exhibit spatial coherence beyond what would be expected from random noise” can you quantify this somehow?

*We added a brief note that the biases are generally spatially correlated to a distance of ~75 km, based on an experimental variogram (Lines ).*

> **“pronounced decrease in accumulation” – how large is this decrease? Is it statistically significant? It looks quite large from Figure S2. Can you see this trend in the firn core?**

*I feel, especially given the space constraints, discussing the significance and source of this decrease is counterproductive. Do you agree?*

> “Site B in this study coincides with a firn core site” – replace with core “location”

*We incorporated the suggested edit (Line ).*

> **Does SEAT2010-4 show any decrease in accumulation between 1990-2016, or is the difference entirely within the radar calculations?**

*Same as the earlier question regarding the decrease in accumulation...*

> **Can you briefly discuss “the limitations on the repeatability of radar imaging in these regions”**

*Once again with the space constraints I don't think it is worth it to go into this, but I'm not sure how to state that to the reviewers. Any thoughts would be helpful.*

> “indicate that the automated PAIPR method is able to accurately replicate the manual approach” – can you quantify the difference in IRH depth between the two methods, not just the accumulation. Might be simple to add a column to one of the tables, since there is also uncertainty in density

*Although we agree this would be potentially useful information, its inclusion with our specific methodology would be more involved than it might first appear.
The PAIPR method first picks as many potential layers as possible and then assigns each layer a likelihood of representing a true annual layer.
It then performs Monte Carlo simulations based on these probabilities to generate age-depth profile distributions for each trace.
No picked layer therefore is absolutely considered an annual layer, making direct comparisons of layer depths to manually picked annual layers rather complicated and non-intuitive.
We feel the necessary context and nuance required to adequately explain this in such a short paper (particularly for an aspect not directly related to the research target) would detract from the overall manuscript as it would require the removal of more pertinent information.*

> “when data collection is separated by a number of years” should be 5 years, since you can’t speculate if this would work with a longer separation

*We have edited this sentence to explicitly refer only to the 5 year separation period.*

> “annual and sub-annual changes in surface conditions” if a regions begins to melt that had previously not experienced melt it would be harder for the snow radar to penetrate that region. So this statement is at least limited to the dry snow zone

*We have edited this sentence to explicitly note that these results are applicable to the dry firn zone of ice sheets specifically (Lines ).*

> “implies the requisite collection frequency is likely much larger” – would you suggest collection every year? Two years? This is where you can make recommendations, especially now that OIB has finished

*There appears to be a slight misunderstanding in our statement.
We conclude that a collection interval of 5 years appears to be sufficient, but some of our evidence supports the notion that longer intervals between data collection are likely acceptable.
We reworded the specific sentence to better reflect this meaning and to avoid confusion (Lines ).*

> Could bring in Koenig’s results to “broadly applicable to dry-firn regions of polar ice sheets in both hemispheres”

*As this is a concluding summary statement, we do not believe this is the correct location to discuss/compare results from other studies.
We have, however, added additional commentary on Koenig et al. (2016) earlier in the manuscript (Lines ).*

> Figure 1 – Fonts are all a bit small and difficult to read. “Mean accum” could be “Mean accumulation.” The thin line indicating mean accumulation is too small to detect variability, especially when on the white background.

*We have increased the size of all fonts and the size of the colored accumulation line to aid in visibility.
We have also modified the color of the thicker line (denoting overlap) to better contrast with the accumulation color mappings.
We have changed "Mean accum" to "Mean accumulation".
We have further incorporated additional modifications to the figure (addition of hillshade and contour lines) to provide better contextual information of the region of study.*

> Figure 2 – I know this is straight from the NSIDC site, but you could Photoshop the text to be a bit larger and easier to read for those less familiar with radar echograms. Can you indicate on Figure 1 where this 5 km transect is?

*We have modified the figure to make the axes and labels easier to read.
We have also expanded this figure to include comparisons between 2011 and 2016 flights for this 5 km transect.
We denote the location of this transect in Fig. 1 with a white star.
This figure now appears in the Supplement due to space constraints.*

> Figure 3 – The legend in Fig 3a. says “2016-2011” but the caption says “2011-2016.” Which is it? Colorbar extends beyond the height of Fig 3b. The colorbar is not white at 0 bias. Colored points are too small to see in Fig 3b. There seems to be a vertical line left of the word “Density”

*We have edited the caption to match the figure (2016-2011).
We have modified the colorbar to match the bounds of the figure.
We have ensured the colorbar is symmetric (although the center color is closer to gray than to white).
We have increased the size of the colored points to increase visibility.
We were unable to discern a vertical line left of the word "Density" in this figure.
This may have been introduced during the conversion after uploading.
If this continues to be an issue, please notify us and we will work to resolve it.*

> The link within the citation for [20] is just for the abstract, not the actual paper

*The full paper can be downloaded from the PDF link on the upper right-hand side of the website.
Nevertheless, we have updated the citation link to point directly to the PDF to avoid confusion.*

> Figure S1 – All fonts are much too small to see. Nowhere in the figure or caption does it say that these numbers are accumulation in mm w.e. a^-1. I’d suggest changing the y axis for the histograms and only having one legend, otherwise it’s difficult to discern differences between the methods.

*We have increased the size of the fonts used, ensured axis labels are included, and made additional changes to increase the readability of the plots in this figure.*

> Figure S2 – Are the dashed lines uncertainty bounds? Need to say so, if true. How common are these missed/extra layers in the PAIPR method? Could you build in a manual correction to prevent this from happening elsewhere?

*We have clarified that the dashed lines are $1\sigma$ uncertainty bounds.
We have additionally clarified and expounded on the missed/extra layers and how these are limited to the manually traced estimates.
The probabilistic nature of PAIPR minimizes the impact of this effect for the automated results.*
