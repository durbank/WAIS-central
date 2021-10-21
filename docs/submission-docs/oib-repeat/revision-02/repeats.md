# Repeated flightlines

This is a list of flightlines that have repeated portions with a sufficient number of years in between for GRSL analysis.
These have not yet been vetted to determine if the repeated portions are of sufficient quality to be used.

## Central WAIS

- 20111109 <==> 20161109
- 20091029 <==> 20141029 <==> 20161104
- 20101120 <==> 20141106
- 20091018 <==> 20091102 <==> 20141121

## West WAIS

- 20091102 <==> 20111113 <==> 20161107

## Ronne Ice Shelf

- 20121018 <==> 20161024

## Priority flights

- Squiggly knob: 20091029 <==> 20161104
- PIG Hook: 20101120 <==> 20141106
- Crossing: 20091018 <==> 20141121
- Concentric bands: 20091102 <==> 20141121

## Response regarding additional flights

Although the entire Operation IceBridge dataset is considerably larger and more expansive than that used in this study, data that meet the necessary requirements for our analysis substantially limits the available data.
The density model used in our study is trained on a specific ice core set with adequate coverage over central WAIS but is more limited over other regions.
Our estimated depth-density profiles are therefore less reliable outside the region of the ice core training data.
As this study is specfically interested in the biases between repeat OIB flights separated over time, we could assume a fixed density profile and still investigate how the differences in collection time affect results.
Even with this assumption however, several additional factors severely limit the fraction of OIB data availabe for use in our study.

First, Snow radar data were collected only on a subset of the OIB flights.
We also require data from flights with substantial repeat coverage separated in time by a number of years (ideally 5 year to match the separation of the current data used in our study).
Repeat flights themselves are somewhat rare, and many of those that do exist were collected only a year or two apart.
We are also limited to portions of the ice sheet where accumulation rates are sufficiently low to enable (within the upper ~25 m) several years of overlapping accumulation years between the repeat flights (ideally 20 years of overlap to match the current data in our study, but a lower number of years could still provide adequate results).
Finally, the annual horizons in the echograms must be sufficiently well-delinated (e.g. the data must be of sufficient quality) to allow for layer picking with a reasonable level of uncertainty.
These requirements necessarily limit our analyses to a subset of the full OIB dataset.

In an effort to address the concerns of the reviewer, we performed a survey of OIB Snow radar data in West Antarctica to search for sufficient portions of additional flights meeting these requirements.
Our survey indicated that our requirements remove the vast majority of the OIB dataset from consideration, with other flights typcially failing to meet one or more of these requirements.
Ultimately, we were able to identify 4 additional pairs of Snow radar flights that met our preliminary requirements (data collection separated by at least 4 years, overlapping flight segments of many 10's of kilometers, and in regions of sufficiently low accumulation rates to potentially yield several years of overlapping annual estimates) but these also failed upon closer inspection.
These pairs are mostly concentrated in central WAIS in the same general region as our current data, but also include repeat flights on the eastern portion of the Ronne Ice Shelf.
The paired flight collection dates are as follows:

- 2009-10-29 <==> 2016-11-04
- 2010-11-20 <==> 2014-11-06
- 2009-10-18 <==> 2014-11-21
- 2012-10-18 <==> 2016-10-24

Many of these flights only contain annually-resolved or well-delineated data over a small portion of the full flight.
Some of these flights do not have any significant portions of sufficiently-well resolved data that overlap for both pairs of flights (as is the case for 2009-10-18/2014-11-21 and 2010-11-20/2014-11-06).
The two remaining candidates also have substantial issues.
Significant portions of the overlapping sections for 2009-10-29/2016-11-04 have sufficiently high accumulation rates that the generated time series are too short to include in our analysis while the estimated uncertainties involved in the remaining portions are unacceptably high.
The small section of overlap for 2012-10-18/2016-10-24 that passes our automated quality control routine (a segment of ~50 km) also exhibits sufficiently high errors (often 2-3 times the magnitude of the estimated accumulation rates) to warrant its exclusion from our analysis.

All of these issues and errors combined lead us to not include additional analyses or data within this manuscript.
It may be possible to extract additional useful information from these in a future study/comparison but such would require a significant investment of additional time and substantial changes to our methodology to do so.
We therefore elect to not include these data in this submission.
We instead have added additional language to our manuscript to better highlight the spatial limits of the data used in the study and include some justification for this more limited data set.
We believe these edits better clarify the limitations of our analysis in this regard.
