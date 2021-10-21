# Response to reviewers

>> While the authors have addressed a wide range of comments and concerns by other reviewers, the limited scope for the data used for their analysis remains problematic. While it's fair that a comprehensive analysis of all OIB data is beyond the scope of this short-format paper, the amount of data used remains troublingly limited.  It's not unreasonable to expect the authors to add lines from one or two more regions to the analysis while still fitting within the size limits of a GRSL paper.

We agree with the reviewer that the paper would be improved with additional data from other regions.
We therefore explored the Operation IceBridge dataset again more carefully to see if we could add more analyses to the current study.
This exploration included a preliminary review of 97 additional OIB flights over West Antarctica.

Although the entire Operation IceBridge dataset is considerably larger and more expansive than that used in this study, the requirements for our analysis substantially limit the available data.
The density model used in our study is trained on a specific ice core set with adequate coverage over central WAIS but more limited over other regions.
Our estimated depth-density profiles are therefore less reliable outside the region of the ice core training data.
As this study is specifically interested in the biases between repeat OIB flights separated over time, we could assume a fixed density profile and still investigate how the differences in collection time affect results.
Even with this assumption however, several additional factors severely limited the fraction of OIB data available for use in our study.

First, Snow radar data were collected only on a subset of the OIB flights.
We also require substantial repeat coverage for flights separated in time by a number of years (ideally 5 years to match the separation of the current data used in our study, although we considered candidates within the range of 4--7 years).
Repeat flights themselves are somewhat rare, and many of those that do exist were collected only a year or two apart.
We are also limited to portions of the ice sheet where accumulation rates are sufficiently low to enable (within the upper ~25 m) several years of overlapping accumulation years between the repeat flights (ideally 20 years of overlap to match the current data in our study, but a lower number of years could still provide adequate results).
Finally, the annual horizons in the echograms must be sufficiently well-delineated (e.g. the data must be of sufficient quality) to allow for layer picking with a reasonable level of certainty.
These requirements necessarily limited our analyses to a subset of the full OIB dataset.

In an effort to address the concerns of the reviewer, we performed a survey of OIB Snow radar data in West Antarctica to search for sufficient portions of additional flights meeting these requirements.
Our survey indicated that our requirements remove the vast majority of the OIB dataset from consideration, with other flights failing to meet one or more of these requirements.
Ultimately, we identified 4 additional pairs of Snow radar flights (of the original 97 surveyed candidates) that met our preliminary requirements (data collection separated by at least 4 years, overlapping flight segments of many 10's of kilometers, and in regions of sufficiently low accumulation rates to potentially yield several years of overlapping annual estimates).
These pairs were mostly concentrated in central WAIS in the same general region as our current data, but also included repeat flights on the eastern portion of the Ronne Ice Shelf.
The paired flight collection dates are as follows:

- 2009-10-29 <==> 2016-11-04
- 2010-11-20 <==> 2014-11-06
- 2009-10-18 <==> 2014-11-21
- 2012-10-18 <==> 2016-10-24

Unfortunately, many of these flights only contained annually-resolved or well-delineated data over a small portion of the full flight.
Some of these flights did not have any significant portions of sufficiently-well resolved data that overlap for both pairs of flights (as is the case for 2009-10-18/2014-11-21 and 2010-11-20/2014-11-06).
The two remaining candidates also had substantial issues.
Significant portions of the overlapping sections for 2009-10-29/2016-11-04 have sufficiently high accumulation rates that the generated overlapping time series are too short (<5 years) to include in our analysis while the estimated uncertainties involved in the remaining portions were unacceptably high.
The small section of overlap for 2012-10-18/2016-10-24 not flagged by our automated quality control routine (a segment of ~50 km) also exhibited errors (often 2-3 times the magnitude of the estimated accumulation rates) too high to warrant inclusion in our analysis.

All of these issues and errors combined lead us to not include additional analyses or data within this manuscript.
It may be possible to extract additional useful information from these in a future study/comparison  with substantial changes to our methodology.
We, therefore, elect to not include these data in this submission.
We instead have added additional language to our manuscript to better highlight the spatial limits of the data used in the study and include some justification for this more limited data set (See Section II in the revised manuscript).
We believe these edits better clarify the limitations of our analysis in this regard.
