## Process for performing sewer capacity caclulations

1. calculate the capacity and travel time in each segment in a network
2. For each segment in a network:
    - accumulate the upstream area
    - identify the longest (tc) upstream path
    - determine the time of concentration along the tc path (sum the travel time of each segment in the tc + 3 minutes)
    - calculate the peak flow via Rational Method
    - calculate the percent over capacity
