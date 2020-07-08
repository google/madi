The Smart Buildings Anomaly Detection dataset consists of 60,425
multidimensional, multimodal observations derived from 15
Variable Air Volume (VAV) climate control devices collected over 14 days
between October 8 and October 21, 2019, from a Google campus in the
California Bay Area. In 1,921 (3.2%) anomalous observations, the devices
are unable to maintain setpoint, and are of interest to facilities
technicians. A setpoint is maintained when the zone air temperature
remains above the zone air heating setpoint, and below the zone air
cooling setpoint. On Mondays through Fridays, from 6:00 am to 10:00 pm
local time, the devices operate in a comfort mode, with tight constraints
between the heating and cooling setpoints. From 10:00 pm to 6:00 am, and
on weekends the setpoints are wider to reduce energy consumption, and
hence, there are comfort and eco operating modes. The data had seven
numeric dimensions: zone air cooling temperature setpoint, zone air
heating temperature setpoint, zone air temperature sensor, supply air
flowrate sensor, supply air damper percentage command, supply air
flowrate setpoint, integer day of week (0-6), integer hour of day (0-23).

Note: This data set is not intended to characterize all possible failure
modes from climate control devices; the anomaly labels represent only one
type of failure mode. However, given this single failure mode, it is useful
to compare the performance of each anomaly detection approach. Future work
should be based on a richer data set that contains  commonly occurring
failure modes on a wider range of climate control devices.
