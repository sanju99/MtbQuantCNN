1. Install snpEff with anaconda: `conda install -c bioconda snpeff`

2. Add the Rv assembly name: Mycobacterium_tuberculosis_gca_000195955 to the snpEff config file:

`anaconda3/envs/bioinformatics/share/snpeff-5.1-2/snpEff.config`

The bacterial codon table must also be specified, instead of the default codon table. So the full text added to the config file is:

```markdown
Mycobacterium_tuberculosis_gca_000195955.genome: Mycobacterium_tuberculosis_gca_000195955

    Mycobacterium_tuberculosis_gca_000195955.reference: https://www.ncbi.nlm.nih.gov/assembly/GCA_000195955.2/
    Mycobacterium_tuberculosis_gca_000195955.retrieval_date: 2023-02-23
    Mycobacterium_tuberculosis_gca_000195955.NC_000962.3.codonTable: Bacterial_and_Plant_Plastid
```

This is based on these <a href="https://pcingola.github.io/SnpEff/se_buildingdb/#configuring-codon-tables-not-always-required" target="_blank">instructions</a>.

3. Create the folder `anaconda3/envs/bioinformatics/share/snpeff-5.1-2/data` and create a folder within that for each database you want to create. The folder name should be the same as the assembly name added to `snpEff.config`. For example:

`anaconda3/envs/bioinformatics/share/snpeff-5.1-2/data/Mycobacterium_tuberculosis_gca_000195955`

Follow the instructions in the <a href="https://pcingola.github.io/SnpEff/se_buildingdb/#step-2-option-2-building-a-database-from-genbank-files" target="_blank">"Step 2, Option 2: Building a database from GenBank files"</a> section of the snpEff instructions for creating a custom database.

1. Download a GenBank file for the H37Rv reference sequence <a href="https://www.ncbi.nlm.nih.gov/nuccore/NC_000962.3" target="_blank">NC_000962.3</a> (EXACTLY follow the instructions in the above link).
2. Because all of our VCF files have been aligned to the NC_000962.3 genome, you HAVE to download this genome. Otherwise, you will get a "Chromosome not found error" when you try to run snpEff on the VCF files.
3. Rename the downloaded file to `genes.gbk` and place in the `anaconda3/envs/bioinformatics/share/snpeff-5.1-2/data/Mycobacterium_tuberculosis_gca_000195955` folder. <b>The name is important because snpEff looks for a file with that exact name.</b>

Run the following to build the database:

```bash
snpEff build -genbank -v Mycobacterium_tuberculosis_gca_000195955
```
To run snpEff on a single file, run

```bash
snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -no-downstream -no-upstream /path/to/sample/file.vcf > /path/to/sample/file.eff.vcf
```

To run it efficiently on many samples without having to reload the database every time, you need to create a text file containing the paths to all the files that you want to annotate. Once you have that file, run:

```bash
snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList -no-downstream -no-upstream /path/to/samples/file.txt
```

<!-- The next block of code extracts the desired fields from each VCF file and saves them to a text file. The text files will then be processed and concatenated into the final `isolate_variants.csv` file.

```bash
paste /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/RIF/paths.txt /home/sak0914/lasso/rif_out_paths.txt | while read vcf_path out_path; do
     bcftools view -v snps,indels "${out_path}.eff.vcf" | bcftools query -i 'INFO/IMPRECISE != 1 & ALT != "."' -f '%POS %REF %ALT %QUAL %FILTER %INFO/DP %INFO/BQ %INFO/MQ %INFO/AF %ANN\n' > "${out_path}.txt"
done
``` -->
