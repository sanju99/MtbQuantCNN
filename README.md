# Convolutional Neural Network to Predict Mtb Drug MICs

## Data Cleaning

Do the following steps in the order for a given drug:

1. Get all isolates with MIC data and full VCF files available in the rollingDB database. 
2. Remove non-CRyPTIC isolates for which only one MIC was tested.
3. Remove isolates with multiple lineages (may have lots of ambiguous calls, or be a mixed sample)
4. Remove isolates with canonical mutations and MICs < 1/2 of the CC. 

## Bounded Loss Function
