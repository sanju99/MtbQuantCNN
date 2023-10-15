#!/bin/bash 
#SBATCH -c 4
#SBATCH -t 1-00:00
#SBATCH -p medium 
#SBATCH --mem=10G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### REQUIRED COMMAND LINE ARGUMENTS (IN THIS ORDER): ####################

# 1. TSV file with 3 columns (AND NO HEADER): 
    # col1: sample ID
    # col2: full path of the FASTQ reads 1 file ZIPPED FILES
    # col3: full path of the FASTQ reads 2 file ZIPPED FILES
# 2. out_dir: output directory where results should be stored. i.e. /n/data1/hms/dbmi/farhat/rollingDB/genomic_data

# set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics # CHANGE TO YOUR OWN ENVIRONMENT OR REMOVE IF RUNNING IN THE BASE ENV


################################################################################################################################################

if ! [ $# -eq 2 ]; then
    echo "Please pass in 2 command line arguments: input file, output_directory"
    exit
fi

# command line arguments
input_file=$1
out_dir=$2
redo_samples="/home/sak0914/MtbQuantCNN/megapipe_rerun.tsv"

repair_script="/home/sak0914/anaconda3/envs/bioinformatics/bin/repair.sh" # CHANGE TO WHATEVER FILE PATH IS IN YOUR ENVIRONMENT
ref_genome="/n/data1/hms/dbmi/farhat/mtb_data/h37rv/h37rv.fna"
min_length=50 # parameter from Max


#################### IF YOU USE THE STANDARD DATABASE, THEN YOU HAVE TO RUN EXTRACT_KRAKEN_READS.PY TO GET ONLY THOSE THAT MAP TO MTBC AND CHILDREN TAXA ####################
#################### IF YOU USE THE MTBC DATABASE, THEN YOU JUST HAVE TO EXTRACT THE CLASSIFIED READS BECAUSE THEY ARE THE ONLY ONES THAT GET CLASSIFIED TO MTBC -- DO THIS!!! ####################

# MTBC database (from Shandu)
kraken_db="/n/data1/hms/dbmi/farhat/mtb_data/kraken/20220922_mtbDB"

# Max previously downloaded this standard Kraken database (contains a lot of organisms)
# kraken_db="/n/data1/hms/dbmi/farhat/mm774/References/Kraken2_DB_Dir/Kraken2_DB"
# extract_kraken_reads_script="/home/sak0914/MtbQuantCNN/data_processing/extract_kraken_reads.py"

# # check that the output directory is in the scratch folder so that nothing in rollingDB is affected by this script (yet)
# if ! grep -q "/n/scratch3" <<< "$out_dir"; then
#     echo "Please specify an output directory in the scratch directory!"
#     exit
# fi

# Check if the output directory exists. If not, create it
if [ ! -d "$out_dir" ]; then
    echo "Creating output directory $out_dir"
    mkdir "$out_dir"
fi

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID fastq1_file fastq2_file
do
    
    # create a directory for each sample to be processed. The sample_ID is the name of the directory 
    # in rollingDB/genomic_data or rollingDB/cryptic_output, each sample has a folder named with the sample name
    sample_out_dir="$out_dir/$sample_ID"
        
    # bam will contain the deduplicated bam files, and pilon will contain the FASTA file and gzipped VCF file
    sample_kraken_dir="$sample_out_dir/kraken"
    sample_bam_dir="$sample_out_dir/bam"
    sample_pilon_dir="$sample_out_dir/pilon"
    subdirs_lst=($sample_kraken_dir $sample_bam_dir $sample_pilon_dir)
    # sample_log_dir="$sample_out_dir/logs"
    # subdirs_lst=($sample_kraken_dir $sample_bam_dir $sample_pilon_dir $sample_log_dir)        
        
    #################################################### STEP 5: REMOVE DUPLICATES USING PICARD ####################################################
        

    # remove duplicates using picard. Save deduplicated bam and bam.bai files in the bam directory!
    picard -Xmx6g MarkDuplicates I="$sample_out_dir/$sample_ID.bam" O="$sample_bam_dir/$sample_ID.dedup.bam" REMOVE_DUPLICATES=true M="$sample_bam_dir/$sample_ID.dedup.bam.metrics" ASSUME_SORT_ORDER=coordinate READ_NAME_REGEX='(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$' || { echo error_occurred=true; }
        
        # for use with post-transition syntax: https://github.com/broadinstitute/picard/wiki/Command-Line-Syntax-Transition-For-Users-(Pre-Transition) 
        # picard -Xmx6g MarkDuplicates -I "$sample_out_dir/$sample_ID.bam" -O "$sample_bam_dir/$sample_ID.dedup.bam" -REMOVE_DUPLICATES true -M "$sample_bam_dir/$sample_ID.dedup.bam.metrics" -ASSUME_SORT_ORDER coordinate -READ_NAME_REGEX '(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$'
    
    # index the deduplicated alignment with samtools, which will create a dedup_bam_file.bai file
    samtools index "$sample_bam_dir/$sample_ID.dedup.bam" || { echo error_occurred=true; }


    #################################################### STEP 6: CALL VARIANTS USING PILON ####################################################


    # the fasta file that pilon creates is the polished version. The goal of pilon is to clean the genome by removing inconsistencies, gaps, etc.
    # But it can also call variants
    # documentation: https://github.com/broadinstitute/pilon/wiki
    # --variant flag will create a VCF file. Otherwise, only a FASTA file is created
    # If you specify the --outdir flag, it will create output FASTA and VCF files with the sample_id prefix in the directory specified after outdir
    #  --minmq 1 --minqual 20 --mindepth 5
    pilon -Xmx18g --genome "$ref_genome" --bam "$sample_bam_dir/$sample_ID.dedup.bam" --output "$sample_ID" --outdir "$sample_pilon_dir" --variant || { echo error_occurred=true; }
    
    # keep all non-reference calls in another VCF file
    bcftools view --types snps,indels,mnps,other "$sample_pilon_dir/$sample_ID.vcf" > "$sample_pilon_dir/${sample_ID}_small.vcf" || { echo error_occurred=true; }

    # then delete the full VCF file and the FASTA file to save space and change the name of the small one
    gzip -c "$sample_pilon_dir/$sample_ID.vcf" > "$sample_pilon_dir/${sample_ID}_full.vcf.gz"
    rm "$sample_pilon_dir/$sample_ID.vcf"
    rm "$sample_pilon_dir/$sample_ID.fasta"
    mv "$sample_pilon_dir/${sample_ID}_small.vcf" "$sample_pilon_dir/$sample_ID.vcf"

    # if any of the steps failed, print an error message and delete the entire directory so that it can be rerun
    # this prevents from the entire job from failing and moves on to the next sample if the current one fails
    #if [ "$error_occurred" = true ]; then
    if [ ! -z "$error_occurred" ]; then

        # Add the sample to the redo file so that it can be rerun
        echo "$sample" >> $redo_samples
    else
        echo $error_occurred
    fi
    
done < "$input_file"