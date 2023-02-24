1. Install snpEff with anaconda: `conda install -c bioconda snpeff`

2. Add the Rv assembly name: Mycobacterium_tuberculosis_gca_000195955 to the snpEff config file:

`anaconda3/envs/sambcftools/share/snpeff-5.1-2/snpEff.config`

3. Create the folder `anaconda3/envs/sambcftools/share/snpeff-5.1-2/data` and create a folder within that for each database you want to create. The folder name should be the same as the assembly name added to `snpEff.config`. For example:

`anaconda3/envs/sambcftools/share/snpeff-5.1-2/data/Mycobacterium_tuberculosis_gca_000195955`

This should have a GenBank file called `genes.gbk`. For this assembly, I downloaded it from https://www.ncbi.nlm.nih.gov/assembly/GCF_000195955.2/

```bash
snpEff build -genbank -v Mycobacterium_tuberculosis_gca_000195955
```

You need to create a text file containing the paths to all the files that you want to annotate. Then can run snpEff on all VCF files in the text file efficiently, without having to reload the database every time. 

```bash
snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_for_annot.txt
```

The next block of code extracts the desired fields from each VCF file and saves them to a text file. The text files will then be processed and concatenated into the final `isolate_variants.csv` file.

```bash
paste /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/RIF/paths.txt /home/sak0914/lasso/rif_out_paths.txt | while read vcf_path out_path; do
     bcftools view -v snps,indels "${out_path}.eff.vcf" | bcftools query -i 'INFO/IMPRECISE != 1 & ALT != "."' -f '%POS %REF %ALT %QUAL %FILTER %INFO/DP %INFO/BQ %INFO/MQ %INFO/AF %ANN\n' > "${out_path}.txt"
done
```

