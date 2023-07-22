#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=30G
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

set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics # CHANGE TO YOUR OWN ENVIRONMENT OR REMOVE IF RUNNING IN THE BASE ENV


################################################################################################################################################

if ! [ $# -eq 2 ]; then
    echo "Please pass in 2 command line arguments: input file, output_directory"
    exit
fi

# command line arguments
input_file=$1
out_dir=$2

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

    # check if the final gzipped VCF file does not exist. Perform the steps only if it doesn't exist
    if [ ! -f "$sample_pilon_dir/${sample_ID}.vcf" ]; then
    
        # Check if the directory exists
        if [ ! -d "$sample_out_dir" ]; then
          # Create the directory if it does not exist
          mkdir "$sample_out_dir"
        fi
        
        # Iterate through the array of subdirectories and create each one if it doesn't exist
        for i in ${!subdirs_lst[@]}; do
            if [ ! -d "${subdirs_lst[$i]}" ]; then
              mkdir -p "${subdirs_lst[$i]}"
            fi
        done
        
        
        ############################################ STEP 1: FIX READ NAMING BETWEEN PAIRED END FILES ###########################################
        
        
        # this is a rare occurrence, but is included for all samples
        # in some FASTQ files, reads are duplicated in one paired end file, so then when bwa tries to align them, it gets a read mismatch
        # it won't get fixed just by sorting the reads because there are multiple of a single read
        # use the repair script from bbmap (conda-installable)
                
        if [ ! -f "$sample_out_dir/$sample_ID.R2.fixed.fastq" ]; then
            /bin/bash $repair_script in="$fastq1_file" in2="$fastq2_file" out="$sample_out_dir/$sample_ID.R1.fixed.fastq" out2="$sample_out_dir/$sample_ID.R2.fixed.fastq" minlen=$min_length
        fi
        
        
        #################################################### STEP 2: TRIM READS USING FASTP ####################################################
        
        
        # trim reads with the minimum allowable read length (post-trimming). This discards reads that are below the minimum length after trimming
        # default is to discard reads in the individual R1/R2 files if only one read in a pair passes QC. 
        # also perform deduplication, as a precaution. Duplicate reads can cause issues with bwa alignment later
        
        if [ ! -f "$sample_out_dir/$sample_ID.R2.fastq" ]; then
            fastp -i "$sample_out_dir/$sample_ID.R1.fixed.fastq" -I "$sample_out_dir/$sample_ID.R2.fixed.fastq" -o "$sample_out_dir/$sample_ID.R1.fastq" -O "$sample_out_dir/$sample_ID.R2.fastq" -h "$sample_out_dir/fastp.html" -j "$sample_out_dir/fastp.json" --length_required $min_length --dedup
        fi
        
        #################################################### STEP 3: ASSIGN READS TO NCBI TAXONOMIC IDS ####################################################
        
        
        # the classified-out flag creates output files with the format following the flag, where # is replaced with _1 or _2 for the paired end reads
        # also pipe the read classification information (which, by default, is printed to stdout) to an output file

        if [ ! -f "$sample_out_dir/${sample_ID}.R_2.kraken.filtered.fastq" ]; then
            kraken2 --db $kraken_db --threads 4 --paired "$sample_out_dir/$sample_ID.R1.fastq" "$sample_out_dir/$sample_ID.R2.fastq" --report "$sample_kraken_dir/kraken_report" --classified-out "$sample_out_dir/${sample_ID}.R#.kraken.filtered.fastq" > "$sample_kraken_dir/kraken_classifications"
        fi
        
        
        #################################################### STEP 4: ALIGN READS TO REFERENCE H37RV GENOME ####################################################
        
        
        # index the reference genome (required before alignment!)
        bwa index $ref_genome

        # align reads to the reference genome sequence. The RG name specifies the read group name. I kept the same format as in previous versions of megapipe
        if [ ! -f "$sample_out_dir/$sample_ID.sam" ]; then
            bwa mem -M -R "@RG\tID:{$sample_ID}\tSM:{$sample_ID}" -t 8 $ref_genome "$sample_out_dir/${sample_ID}.R_1.kraken.filtered.fastq" "$sample_out_dir/${sample_ID}.R_2.kraken.filtered.fastq" > "$sample_out_dir/$sample_ID.sam"
        fi
        
        # sort alignment and convert to bam file
        if [ ! -f "$sample_out_dir/$sample_ID.bam" ]; then
            samtools view -bS "$sample_out_dir/$sample_ID.sam" -m 4G | samtools sort -m 4G > "$sample_out_dir/$sample_ID.bam"
        fi
        
        # index alignment, which creates a .bai index file
        if [ ! -f "$sample_out_dir/$sample_ID.bam.bai" ]; then
            samtools index "$sample_out_dir/$sample_ID.bam"
        fi
        
        
        #################################################### STEP 5: REMOVE DUPLICATES USING PICARD ####################################################
        

        # remove duplicates using picard. Save deduplicated bam and bam.bai files in the bam directory!
        if [ ! -f "$sample_bam_dir/$sample_ID.dedup.bam" ]; then
            picard -Xmx6g MarkDuplicates I="$sample_out_dir/$sample_ID.bam" O="$sample_bam_dir/$sample_ID.dedup.bam" REMOVE_DUPLICATES=true M="$sample_bam_dir/$sample_ID.dedup.bam.metrics" ASSUME_SORT_ORDER=coordinate
        fi
            # for use with post-transition syntax: https://github.com/broadinstitute/picard/wiki/Command-Line-Syntax-Transition-For-Users-(Pre-Transition) 
            # picard -Xmx6g MarkDuplicates -I "$sample_out_dir/$sample_ID.bam" -O "$sample_bam_dir/$sample_ID.dedup.bam" -REMOVE_DUPLICATES true -M "$sample_bam_dir/$sample_ID.dedup.bam.metrics" -ASSUME_SORT_ORDER coordinate
        
        # index the deduplicated alignment with samtools, which will create a dedup_bam_file.bai file
        if [ ! -f "$sample_bam_dir/$sample_ID.dedup.bam.bai" ]; then
            samtools index "$sample_bam_dir/$sample_ID.dedup.bam"
        fi


        #################################################### STEP 6: CALL VARIANTS USING PILON ####################################################


        # the fasta file that pilon creates is the polished version. The goal of pilon is to clean the genome by removing inconsistencies, gaps, etc.
        # But it can also call variants
        # documentation: https://github.com/broadinstitute/pilon/wiki
        # --variant flag will create a VCF file. Otherwise, only a FASTA file is created
        # If you specify the --outdir flag, it will create output FASTA and VCF files with the sample_id prefix in the directory specified after outdir
        if [ ! -f "$sample_pilon_dir/$sample_ID.fasta" ]; then
            pilon -Xmx18g --genome "$ref_genome" --bam "$sample_bam_dir/$sample_ID.dedup.bam" --output "$sample_ID" --outdir "$sample_pilon_dir" --variant
        fi
        
        # keep all non-reference calls in another VCF file
        bcftools view --types snps,indels,mnps,other "$sample_pilon_dir/$sample_ID.vcf" > "$sample_pilon_dir/${sample_ID}_small.vcf"

        # then delete the full VCF file and the FASTA file to save space and change the name of the small one
        rm "$sample_pilon_dir/$sample_ID.vcf"
        rm "$sample_pilon_dir/$sample_ID.fasta"
        mv "$sample_pilon_dir/${sample_ID}_small.vcf" "$sample_pilon_dir/$sample_ID.vcf"

    else
        echo "$sample_pilon_dir/${sample_ID}.vcf already exists"
    fi

    # everything that's needed is in the bam, kraken, or pilon directories, so everything in sample_out_dir can be deleted for space in this version
    # set maxdepth to 1 so that only files in this directory will be deleted, it will not search in subdirectories
    # -type f means only files
    find $sample_out_dir -maxdepth 1 -type f -delete
    
    
done < "$input_file"