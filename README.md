# MtbQuantCNN

## Data Cleaning

Do the following steps in the order:

1. Remove non-CRyPTIC isolates for which only one MIC was tested.
2. Remove isolates with multiple lineages (may have lots of ambiguous calls, or be a mixed sample)
3. Remove isolates with canonical mutations and MICs < 0.5 * CC. 

## Bounded Loss Function
