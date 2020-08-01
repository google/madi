# Multidimensional multimodal Anomaly Detection with Interpretation (MADI)

This is the source code that accompanies
Iterpretable, Multidimensional, Multimodal Anomaly
Detection for Detecting Device Failure (Sipple, 2020)

## Abstract
In this paper we propose a scalable, unsupervised
approach for detecting anomalies in the Internet
of Things (IoT). Complex devices are connected
daily and eagerly generate vast streams of multidi-
mensional telemetry. These devices often operate
in distinct modes based on external conditions
(day/night, occupied/vacant, etc.), and to prevent
complete or partial system outage, we would like
to recognize as early as possible when these de-
vices begin to operate outside the normal modes.

We propose an unsupervised anomaly detection
method that creates a negative sample from the
positive, observed sample, and trains a classifier
to distinguish between positive and negative sam-
ples. Using the Concentration Phenomenon, we
explain why such a classifier ought to establish
suitable decision boundaries between normal and
anomalous regions, and show how Integrated Gra-
dients can attribute the anomaly to specific dimen-
sions within the anomalous state vector. We have
demonstrated that negative sampling with random
forest or neural network classifiers yield signifi-
cantly higher AUC scores compared to state-of-
the-art approaches against benchmark anomaly
detection datasets, and a multidimensional, multi-
modal dataset from real climate control devices.

Finally, we describe how negative sampling with
neural network classifiers have been successfully
deployed at large scale to predict failures in real
time in over 15,000 climate-control and power
meter devices in 145 office buildings within the
California Bay Area.

