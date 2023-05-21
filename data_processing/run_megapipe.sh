#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-01:00
#SBATCH -p short 
#SBATCH --mem=25G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### REQUIRED COMMAND LINE ARGUMENTS (IN THIS ORDER): ####################

# 1. TSV file with 3 columns: 
    # col1: sample ID
    # col2: full path of the FASTQ reads 1 file
    # col3: full path of the FASTQ reads 2 file
# 2. out_dir: output directory where results should be stored. i.e. /n/data1/hms/dbmi/farhat/rollingDB/genomic_data
# 3. kraken_filtering: string "true" or "false" denoting if kraken filtering should be performed. Can be any case, it will get converted to all uppercase

set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior


################################################################################################################################################

# PLEASE NOTE THAT THIS SCRIPT IS NOT INTENDED FOR GENERAL LAB USE. IT WRITES FILES ONLY TO THE SCRATCH DIRECTORY AND DELETES MOST INTERMEDIATE FILES
# I AM JUST USING THIS TO PROCESS VCF FILES ASAP FOR MY WORK. BUT THE COMMANDS ARE ALL GOOD AND CAN BE USED ELSEWHERE

################################################################################################################################################


if ! [ $# -eq 3 ]; then
    echo "Please pass in 3 command line arguments: input file, output_directory, and the full path to the bbmap repair script"
    exit
fi

input_file=$1

# /n/scratch3/users/s/sak0914/MIC_ML
out_dir=$2

# /home/sak0914/anaconda3/envs/bioinformatics/bin/repair.sh
repair_script=$3

# extract_kraken_reads=${3^^}
ref_genome="/n/data1/hms/dbmi/farhat/mtb_data/h37rv/h37rv.fna"
min_length=50

# Max previously downloaded the standard Kraken database to /n/data1/hms/dbmi/farhat/mm774/References/Kraken2_DB_Dir/Kraken2_DB
# kraken_db="/n/data1/hms/dbmi/farhat/mm774/References/Kraken2_DB_Dir/Kraken2_DB"

#################### IF YOU USE THE STANDARD DATABASE, THEN YOU HAVE TO RUN EXTRACT_KRAKEN_READS.PY TO GET ONLY THOSE THAT MAP TO MTBC AND CHILDREN TAXA ####################
#################### IF YOU USE THE MTBC DATABASE, THEN YOU JUST HAVE TO EXTRACT THE CLASSIFIED READS BECAUSE THEY ARE THE ONLY ONES THAT GET CLASSIFIED TO MTBC ####################

kraken_db="/n/data1/hms/dbmi/farhat/mtb_data/kraken/20220922_mtbDB"
extract_kraken_reads_script="/home/sak0914/MtbQuantCNN/data_processing/extract_kraken_reads.py"

# check that the output directory is in the scratch folder so that nothing in rollingDB is affected by this script (yet)
if ! grep -q "/n/scratch3" <<< "$out_dir"; then
    echo "Please specify an output directory in the scratch directory!"
    exit
fi

# Check if the output directory exists. If not, create it
if [ ! -d "$out_dir" ]; then
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
    if [ ! -f "$sample_pilon_dir/${sample_ID}_full.vcf.gz" ]; then
    
        # Check if the directory exists
        if [ ! -d "$sample_out_dir" ]; then
          # Create the directory if it does not exist
          mkdir "$sample_out_dir"
        fi
        # # raise an error here once you switch to writing output files to rollingDB. Don't want to overwrite anything that's already there
        # # else
        #   # exit
        # fi
        
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
        # documentation: https://github.com/broadinstitute/pilon/wiki
        # If you specify the --outdir flag, it will create output FASTA and VCF files with the sample_id prefix in the directory specified after outdir
        if [ ! -f "$sample_pilon_dir/$sample_ID.fasta" ]; then
            pilon -Xmx18g --genome "$ref_genome" --bam "$sample_bam_dir/$sample_ID.dedup.bam" --output "$sample_ID" --outdir "$sample_pilon_dir" --variant
        fi
        
        # create a VCF file using freebayes
        freebayes -f "$sample_pilon_dir/$sample_ID.fasta" "$sample_bam_dir/$sample_ID.dedup.bam" > "$sample_pilon_dir/$sample_ID.freebayes.vcf"
        
        # gzip the file. -c flag directs the output to stdout. Then delete the unzipped version
        gzip -c < "$sample_pilon_dir/$sample_ID.vcf" > "$sample_pilon_dir/${sample_ID}_full.vcf.gz"
        #rm "$sample_pilon_dir/$sample_ID.vcf"
        # rm "$sample_pilon_dir/$sample_ID.fasta"
        
        # keep all variants in the rpoBC and gyrBA regions used for the models and write to new files
        # bcftools view --types snps,indels,mnps,other "$sample_pilon_dir/${sample_ID}_full.vcf.gz" | bcftools filter -i "(POS >= 4998 & POS <= 9818) | (POS >= 759611 & POS <= 767320)" > "/n/scratch3/users/s/sak0914/RIF_MXF_validation_data/$sample_ID.vcf"

        # keep all non-reference calls in another VCF file
        bcftools view --types snps,indels,mnps,other "$sample_pilon_dir/${sample_ID}_full.vcf.gz" > "$sample_pilon_dir/${sample_ID}.vcf"

    else
        echo "$sample_pilon_dir/${sample_ID}_full.vcf.gz exists"
    fi

    # everything that's needed is in the bam or pilon directories, so everything in sample_out_dir can be deleted for space in this version
    # IF/WHEN THIS IS ADAPTED FOR GENERAL LAB USE, WANT TO KEEP SOME FILES
    # set maxdepth to 1 so that only files in this directory will be deleted, it will not search in subdirectories
    # -type f means only files
    find $sample_out_dir -maxdepth 1 -type f -delete
    
    
done < "$input_file"