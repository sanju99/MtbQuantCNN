#!/bin/bash 
#SBATCH -c 8
#SBATCH -t 0-06:00
#SBATCH -p short 
#SBATCH --mem=30G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### REQUIRED COMMAND LINE ARGUMENTS (IN THIS ORDER): ####################

# 1. TSV file with 2 columns (AND NO HEADER): 
    # col1: sample_ID, i.e. "SAMEA104362063"
    # col2: comma-separated string of all sequencing runs for the sample in col1, i.e. "ERR2184222,ERR9029914"
# 2. fastq_dir: directory where paired-end Illumina sequencing reads are stored
# 3. out_dir: output directory where results should be stored. i.e. /n/data1/hms/dbmi/farhat/rollingDB/cryptic_output or /n/data1/hms/dbmi/farhat/rollingDB/genomic_data

##### NOTE: The values in col1 and col2 of the input file can be the same, i.e. for unpublished data. But for standardization, it is best to use the BioSample accession ID in col1 and sequencing run IDs in col2 #####

set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics # CHANGE TO YOUR OWN ENVIRONMENT OR REMOVE IF RUNNING IN THE BASE ENV

if ! [ $# -eq 3 ]; then
    echo "Please pass in 3 command line arguments: a text file with two columns: sample_ID and sequencing runs, the FASTQ directory, and the output directory"
    exit
fi

# command line arguments
input_file=$1
fastq_dir=$2
out_dir=$3 # directory where completed variant calling results for sample_ID IDs will be stored (OUTPUT)

repair_script="/home/sak0914/anaconda3/envs/bioinformatics/bin/repair.sh" # CHANGE TO WHATEVER FILE PATH IS IN YOUR ENVIRONMENT
ref_genome="/n/data1/hms/dbmi/farhat/mtb_data/h37rv/h37rv.fna"
genome_length=$(tail -n +2 $ref_genome | tr -d '\n' | wc -c) # remove first line (FASTA header) and newline characters, then count characters to get ref genome length

min_length=50 # parameter determined by Max. Drop trimmed reads that are shorter than this length

#################### IF YOU USE THE STANDARD DATABASE, THEN YOU HAVE TO RUN EXTRACT_KRAKEN_READS.PY TO GET ONLY THOSE THAT MAP TO MTBC AND CHILDREN TAXA ####################
#################### IF YOU USE THE MTBC DATABASE, THEN YOU JUST HAVE TO EXTRACT THE CLASSIFIED READS BECAUSE THEY ARE THE ONLY ONES THAT GET CLASSIFIED TO MTBC -- DO THIS!!! ####################

# custom MTBC database
kraken_db="/n/data1/hms/dbmi/farhat/mtb_data/kraken/20220922_mtbDB"
kraken_unclassified_max=20
redo_file="./redo_files.txt"
annot_fName="./fNames_for_annot.txt"

# if the file doesn't exist, create empty file, so wc -l will return 0
touch $annot_fName

# empty it (which is necessary if the file already existed before running touch)
truncate -s 0 $annot_fName

# Max previously downloaded this standard Kraken database (contains a lot of organisms)
# kraken_db="/n/data1/hms/dbmi/farhat/mm774/References/Kraken2_DB_Dir/Kraken2_DB"
# extract_kraken_reads_script="./MtbQuantCNN/data_cleaning/extract_kraken_reads.py"
lineage_def_ref_pos="./MtbQuantCNN/data_processing/variant_calling/Coll2014_positions_all.txt" # for F2 calculation

# exclude_regions_file="/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/RLC_Regions.Plus.LowPmapK50E4.H37Rv.ExtendSingleNucs.bed" # from Marin et al., 2022
# this file is directory from the paper's Github repo: /n/data1/hms/dbmi/farhat/Sanjana/H37Rv/RLC_Regions.Plus.LowPmapK50E4.H37Rv.bed
# to get ExtendSingleNucs, I extended each region that is only 1 bp long by 5 bp in both directions 

# Check if the output directory exists. If not, raise an error
if [ ! -d "$out_dir" ]; then
    echo "Output directory $out_dir doesn't exist!"
    exit 1
fi

# define function to check FASTQ file quality
function check_FASTQ_quality() {
    
    local run_ID="$1"

    # FASTQ files for the single sequencing run
    fastq1_file="$fastq_dir/${run_ID}_R1.fastq.gz"
    fastq2_file="$fastq_dir/${run_ID}_R2.fastq.gz"
            
    # unzip FASTQ files to compare
    FQ1_unzipped="${fastq1_file/.gz/}"
    FQ2_unzipped="${fastq2_file/.gz/}"

    gunzip -c $fastq1_file > $FQ1_unzipped
    gunzip -c $fastq2_file > $FQ2_unzipped

    # first check that the original FASTQ files have the same numbers of lines
    FQ1_line_count=$(wc -l $FQ1_unzipped | awk '{print $1}')
    FQ2_line_count=$(wc -l $FQ2_unzipped | awk '{print $1}')
    
    # check that neither FASTQ file has no reads
    if [ $FQ1_line_count -eq 0 ] || [ $FQ2_line_count -eq 0 ]; then
        echo "Error: $fastq1_file or $fastq2_file has no reads"
        exit 1
    # Compare the counts and raise an error if they are not equal 
    elif [ "$FQ1_line_count" -ne "$FQ2_line_count" ]; then
        echo "Error: $fastq1_file and $fastq2_file have different line counts: $FQ1_line_count and $FQ2_line_count"
        exit 1
    else
        echo "Line counts in paired-end FASTQ files for $run_ID match"
    fi

    # compare paired end read files. If they are the same, then add to error list. Suppress output with -s tag, so it doesn't print out the differences
    # If the files are identical, it returns an exit status of 0, and the condition is considered true, so an error will be returned.
    if cmp -s "$FQ1_unzipped" "$FQ2_unzipped"; then
       echo "Error: $fastq1_file and $fastq2_file are duplicates"
       exit 1
    fi

    # then delete the unzipped files
    rm $FQ1_unzipped
    rm $FQ2_unzipped
}



function make_subdirs() {

    local run_out_dir="$1"
    
    # bam will contain the deduplicated bam files, and pilon will contain the FASTA file and gzipped VCF file
    run_fastp_dir="$run_out_dir/fastp"
    run_kraken_dir="$run_out_dir/kraken"
    run_bam_dir="$run_out_dir/bam"
    
    subdirs_lst=($run_fastp_dir $run_kraken_dir $run_bam_dir)
    
    # Iterate through the array of subdirectories and create each one if it doesn't exist
    # the -p flag creates all nested directories if they don't exist
    for i in ${!subdirs_lst[@]}; do
        if [ ! -d "${subdirs_lst[$i]}" ]; then
          mkdir -p "${subdirs_lst[$i]}"
        fi
    done

}



# define function to perform the trimming, kraken-classification, and aligning steps
function repair_trim_classify_reads() {

    local run_ID="$1"
    local run_out_dir="$2"

    local run_fastp_dir="$run_out_dir/fastp"
    local run_kraken_dir="$run_out_dir/kraken"

    # FASTQ files for the single sequencing run
    fastq1_file="$fastq_dir/${run_ID}_R1.fastq.gz"
    fastq2_file="$fastq_dir/${run_ID}_R2.fastq.gz"

    # this is a rare occurrence, but is included for all samples
    # in some FASTQ files, reads are duplicated in one paired end file, so then when bwa tries to align them, it gets a read mismatch
    # it won't get fixed just by sorting the reads because there are multiple of a single read
    # use the repair script from bbmap (conda-installable)
    if [ ! -f "$run_out_dir/$run_ID.R2.fixed.fastq" ]; then
        /bin/bash $repair_script in="$fastq1_file" in2="$fastq2_file" out="$run_out_dir/$run_ID.R1.fixed.fastq" out2="$run_out_dir/$run_ID.R2.fixed.fastq"
    fi    
    
    # trim reads with the minimum allowable read length (post-trimming). This discards reads that are below the minimum length after trimming
    # default is to discard reads in the individual R1/R2 files if only one read in a pair passes QC. 
    # also perform deduplication, as a precaution. Duplicate reads can cause issues with bwa alignment later
    if [ ! -f "$run_out_dir/$run_ID.R2.fastq" ]; then
        fastp -i "$run_out_dir/$run_ID.R1.fixed.fastq" -I "$run_out_dir/$run_ID.R2.fixed.fastq" -o "$run_out_dir/$run_ID.R1.fastq" -O "$run_out_dir/$run_ID.R2.fastq" -h "$run_fastp_dir/fastp.html" -j "$run_fastp_dir/fastp.json" --length_required $min_length --dedup --thread 8
    fi

    # the classified-out flag creates output files with the format following the flag, where # is replaced with _1 or _2 for the paired end reads
    # also pipe the read classification information (which, by default, is printed to stdout) to an output file
    if [ ! -f "$run_out_dir/${run_ID}.R_2.kraken.filtered.fastq" ]; then
        kraken2 --db $kraken_db --threads 8 --paired "$run_out_dir/$run_ID.R1.fastq" "$run_out_dir/$run_ID.R2.fastq" --report "$run_kraken_dir/kraken_report" --classified-out "$run_out_dir/$run_ID.R#.kraken.filtered.fastq" > "$run_out_dir/kraken_classifications"
    fi

}



function align_reads_drop_duplicates() {

    local run_ID="$1"
    local run_out_dir="$2"

    local run_kraken_dir="$run_out_dir/kraken"
    local run_bam_dir="$run_out_dir/bam"

    if [ -f "$run_bam_dir/$run_ID.dedup.bam" ]; then
        echo "$run_bam_dir/$run_ID.dedup.bam was already created"
        
    else
    
        # get unclassified percentage. If it's more than 10%, then skip the rest of the steps to save time
        # kraken percentages are out of 100, so check if the value is less than or equal to 10
        unclassified_percent=$(cat "$run_kraken_dir/kraken_report" | grep unclassified  | awk '{print $1}')
        echo "$run_ID non-MTBC read percentage: $unclassified_percent%"
    
        if [ "$(awk 'BEGIN{print ('$unclassified_percent' <= '$kraken_unclassified_max')}')" -eq 1 ]; then

            echo "Aligning reads to the reference genome"
            
            # index the reference genome (required before alignment!). Converted to using bwa-mem2 to try to reduce segmentation faults. 
            # It's also ~2x faster than bwa-mem: https://arxiv.org/abs/1907.12931
            if [ ! -f "$ref_genome.ann" ]; then # this is one of the outputs of bwa index. there are several, just check for the presence of one.
                bwa index $ref_genome
            fi
            # bwa-mem2 index $ref_genome
    
            # align reads to the reference genome sequence. The RG name specifies the read group name. I kept the same format as in previous versions of megapipe
            if [ ! -f "$run_out_dir/$run_ID.sam" ]; then
                bwa mem -M -R "@RG\tID:{$run_ID}\tSM:{$run_ID}" -t 8 $ref_genome "$run_out_dir/$run_ID.R_1.kraken.filtered.fastq" "$run_out_dir/$run_ID.R_2.kraken.filtered.fastq" > "$run_out_dir/$run_ID.sam"
                # bwa-mem2 mem -M -R "@RG\tID:{$run_ID}\tSM:{$run_ID}" -t 8 $ref_genome "$run_out_dir/$run_ID.R_1.kraken.filtered.fastq" "$run_out_dir/$run_ID.R_2.kraken.filtered.fastq" > "$run_out_dir/$run_ID.sam"
            fi
            
            # sort alignment and convert to bam file
            if [ ! -f "$run_out_dir/$run_ID.bam" ]; then
                samtools view -b "$run_out_dir/$run_ID.sam" -m 4G | samtools sort -m 4G > "$run_out_dir/$run_ID.bam"
            fi
            
            # index alignment, which creates a .bai index file
            if [ ! -f "$run_out_dir/$run_ID.bam.bai" ]; then
                samtools index "$run_out_dir/$run_ID.bam"
            fi        
    
            # remove duplicates using picard. Save deduplicated bam and bam.bai files in the bam directory!
            # setting REMOVE_DUPLICATES = True will remove all duplicates: both those from the sequencing process (PCR duplicates) and the library preparation process
            if [ ! -f "$run_bam_dir/$run_ID.dedup.bam" ]; then
                # -Xmx6g specifies to allocate 6 GB
                picard -Xmx30g MarkDuplicates I="$run_out_dir/$run_ID.bam" O="$run_bam_dir/$run_ID.dedup.bam" REMOVE_DUPLICATES=true M="$run_bam_dir/$run_ID.dedup.bam.metrics" ASSUME_SORT_ORDER=coordinate READ_NAME_REGEX='(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$'
            fi
                # for use with post-transition syntax: https://github.com/broadinstitute/picard/wiki/Command-Line-Syntax-Transition-For-Users-(Pre-Transition) 
                # picard -Xmx6g MarkDuplicates -I "$run_out_dir/$run_ID.bam" -O "$run_bam_dir/$run_ID.dedup.bam" -REMOVE_DUPLICATES true -M "$run_bam_dir/$run_ID.dedup.bam.metrics" -ASSUME_SORT_ORDER coordinate -READ_NAME_REGEX '(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$'
            
            # index the deduplicated alignment with samtools, which will create a dedup_bam_file.bai file
            if [ ! -f "$run_bam_dir/$run_ID.dedup.bam.bai" ]; then
                samtools index "$run_bam_dir/$run_ID.dedup.bam"
            fi
    
            # check if bam file is extremely small (i.e. less than 10 reads), then redo the sample
            bam_line_count=$(wc -l "$run_bam_dir/$run_ID.dedup.bam" | awk '{print $1}')
            
            if [ $bam_line_count -lt 10 ]; then
                echo "Bam file $run_bam_dir/$run_ID.dedup.bam contains too few lines: $bam_line_count"
                echo "$run_ID" >> $redo_file
            fi
    
        fi

    fi
}

    
# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID run_IDs_string
do

    echo "Starting variant calling pipeline for $sample_ID"

    # create a directory for each sample to be processed. The sample_ID is the name of the directory 
    # in rollingDB/genomic_data or rollingDB/cryptic_output, each sample has a folder named with the sample name
    # bam dir contain the deduplicated bam files, pilon dir will contain the VCF files, and F2 dir will contain the .txt file with the F2 metric 
    sample_out_dir="$out_dir/$sample_ID"

    # make top-level directory first
    if [ ! -d "$sample_out_dir" ]; then
        mkdir "$sample_out_dir"
    fi

    sample_bam_dir="$sample_out_dir/bam"
    sample_pilon_dir="$sample_out_dir/pilon"
    sample_lineage_dir="$sample_out_dir/lineage_calling"

    # Iterate through the array of subdirectories and create each one if it doesn't exist
    subdirs_lst=($sample_bam_dir $sample_pilon_dir $sample_lineage_dir)
    
    # the -p flag creates all nested directories if they don't exist
    for i in ${!subdirs_lst[@]}; do
        if [ ! -d "${subdirs_lst[$i]}" ]; then
          mkdir -p "${subdirs_lst[$i]}"
        fi
    done

    # check if a sample is finished. Both of these files must be present for it to have been completed
    if [ ! -f "$sample_pilon_dir/${sample_ID}_full.vcf.gz" ] || [ ! -f "$sample_pilon_dir/${sample_ID}_variants.vcf" ]; then

        # if they are the same, don't make an extra directory level for sequencing runs
        if [[ "$sample_ID" == "$run_IDs_string" ]]; then
    
            # make subdirectories for the sequencing run
            make_subdirs $sample_out_dir
    
            if [ ! -f "$sample_out_dir/kraken/kraken_report" ]; then
    
                # first check FASTQ quality using the function. The function will return an error if conditions are not met
                # check_FASTQ_quality $sample_ID
                    
                repair_trim_classify_reads $sample_ID $sample_out_dir
            
            # also check that the kraken_report is not empty. -s $fName indicates not empty
            else
                if [ -s "$sample_out_dir/kraken/kraken_report" ]; then
                    echo "Kraken report for $sample_ID already exists"
                else
                    echo "Kraken report for $sample_ID is empty"
    
                    # make two columns in the output file so that you can keep track of both the sample ID and the run ID (in this case, they are the same)
                    echo "$sample_ID\t$sample_ID" >> $redo_file
                fi
            fi

            align_reads_drop_duplicates $sample_ID $sample_out_dir
            
            # delete all files that are not in the subdirectories for each sequencing run. these files are not critical, so it saves space
            find $sample_out_dir -maxdepth 1 -type f -delete

            # create an empty file first (because the file needs to exist for the subsequent steps, and if the bam file doesn't exist, then run_IDs.txt will be empty)
            touch "$sample_bam_dir/run_IDs.txt"

            # add the sample ID to the list of bams to include if the bam exists (which is only if the run passes kraken classification)
            if [ -f "$out_dir/$sample_ID/bam/$sample_ID.dedup.bam" ]; then
                echo "$out_dir/$sample_ID/bam/$sample_ID.dedup.bam" > "$sample_bam_dir/run_IDs.txt"
            fi
            
        else
        
            # create an array of the sequencing runs associated with a single isolate
            IFS=',' read -ra run_IDs_array <<< "$run_IDs_string"
            
            # within each sample directory, create a directory for all individual sequencing runs
            for run_ID in "${run_IDs_array[@]}"; do
    
                # make top-level directory first
                run_out_dir="$sample_out_dir/$run_ID"
                
                if [ ! -d "$run_out_dir" ]; then
                    mkdir "$run_out_dir"
                fi
    
                # make subdirectories for each sequencing run
                make_subdirs $run_out_dir

                # perform trimming, kraken-classification, and read alignment for each sequencing run
                if [ ! -f "$run_out_dir/kraken/kraken_report" ]; then
    
                    # first check FASTQ quality using the function. The function will return an error if conditions are not met
                    # check_FASTQ_quality $run_ID
                        
                    repair_trim_classify_reads $run_ID $run_out_dir
                    
                # also check that the kraken_report is not empty. -s $fName indicates not empty
                else
                    if [ -s "$run_out_dir/kraken/kraken_report" ]; then
                        echo "Kraken report for $run_ID already exists"
                    else
                        echo "Kraken report for $run_ID is empty"
    
                        # make two columns in the output file so that you can keep track of both the sample ID and the run ID
                        echo "$sample_ID\t$run_ID" >> $redo_file
                    fi
                fi

                align_reads_drop_duplicates $run_ID $run_out_dir
        
                # delete all files that are not in the subdirectories for each sequencing run. these files are not critical, so it saves space
                find $run_out_dir -maxdepth 1 -type f -delete
    
            done

            # create file of all bams full names
            for run_ID in "${run_IDs_array[@]}"; do
        
                # add the run ID to the list of bams to include if the bam exists (which is only if the run passes kraken classification)
                if [ -f "$out_dir/$sample_ID/$run_ID/bam/$run_ID.dedup.bam" ]; then
                    echo "$out_dir/$sample_ID/$run_ID/bam/$run_ID.dedup.bam"
                fi
                
            done > "$sample_bam_dir/run_IDs.txt"
                
        fi
    
        num_run_IDs=$(wc -l "$sample_bam_dir/run_IDs.txt" | awk '{print $1}')
    
        # check if run_IDs.txt is empty. If not, that means all sequencing runs failed kraken classification, so there are no BAM files for this sample
        # in the for loop above, run_IDs.txt is still created, so have to check if there are no lines, not just file presence vs. absence
        if [ $num_run_IDs -eq 0 ]; then
            echo "No BAM files created for $sample_ID because they all failed kraken classification"

        else
        
            # get all runs associated with this sample_ID and compute depth
            # -a computes depth at all positions, not just those with non-zero depth
            # -Q is for minimum mapping quality: use 1, so that multiply mapped reads aren't counted. These have mapping quality of 0
            if [ ! -f "$sample_bam_dir/$sample_ID.tsv" ]; then
                if [ -f "$sample_bam_dir/$sample_ID.depth.tsv.gz" ]; then
                    # unzip to use in the next steps
                    gunzip "$sample_bam_dir/$sample_ID.depth.tsv.gz"
                else
                    samtools depth -a -Q 1 -f "$sample_bam_dir/run_IDs.txt" > "$sample_bam_dir/$sample_ID.depth.tsv"
                fi
            fi
            
            # when there are multiple bam files, each one is its own column in the depth file.
            num_sites_H37Rv=$(wc -l "$sample_bam_dir/$sample_ID.depth.tsv" | awk '{print $1}')
        
            if [ ! $num_sites_H37Rv -eq $genome_length ]; then
                echo "Check that all $genome_length sites in the H37Rv reference genome are in $sample_bam_dir/$sample_ID.depth.tsv, which currently has $num_sites_H37Rv sites"
                # exit 1

                # delete and remake it if not the correct length
                rm "$sample_bam_dir/$sample_ID.depth.tsv"                
                samtools depth -a -Q 1 -f "$sample_bam_dir/run_IDs.txt" > "$sample_bam_dir/$sample_ID.depth.tsv"
                
            fi
        
            # easier to do this in Python: compute median depth and number of sites with at least 20x coverage
            # only include the bam file for a given run if the median depth is > 15x at least 95% of the Mtb genome has a a depth of at least 20x
            if [ ! -f "$sample_bam_dir/pass_run_IDs.txt" ]; then
                python3 -u ./MtbQuantCNN/data_processing/variant_calling/BAM_depth_QC.py "$sample_out_dir"
            fi

            # process all sequences, not just those that meet the BAM depth threshold. The script above creates pass_run_IDs.txt, but overwrite it here so that all BAMs are included
            cp "$sample_bam_dir/run_IDs.txt" "$sample_bam_dir/pass_run_IDs.txt"

            # then gzip the depth file to save space
            if [ -f "$sample_bam_dir/$sample_ID.depth.tsv" ]; then
                gzip -f "$sample_bam_dir/$sample_ID.depth.tsv"
            fi
            
            num_runs_passed=$(wc -l "$sample_bam_dir/pass_run_IDs.txt" | awk '{print $1}')
            
            # stop processing samples that don't pass the BAM coverage requirements
            if [ $num_runs_passed -eq 0 ]; then
                echo "No BAM files for $sample_ID passed the minimum coverage requirements"
                
            else 
                # if only one BAM file passed, or there is only one sequencing run for this isolate, just use that BAM file for variant calling (DON'T COPY IT BECAUSE THAT USES UP UNNECESSARY SPACE!)
                if [ $num_runs_passed -eq 1 ]; then
        
                    # this file was also already indexed in trim_classify_align, so it doesn't need to be done again
                    bam_file=$(cat "$sample_bam_dir/pass_run_IDs.txt")

                    # TODO: Move the BAM file to the sample_bam_dir and rename it with the sample ID so that it's easier to find that having to go into run_bam_dir
        
                # only remaining case is num_runs_passed is greater than 1
                else
                    echo "$num_runs_passed BAM files passed the minimum coverage requirements" 
        
                    # merged bam file
                    bam_file="$sample_bam_dir/$sample_ID.dedup.bam"
            
                    # merge them using samtools. -f means to overwrite file if it already exists, which is not enabled by default
                    # original bam files were sorted prior to running picard and dropping duplicates (after which they remain sorted), so samtools merge will work (required sorted bam files)
                    if [ ! -f $bam_file ]; then
                        samtools merge -f -b "$sample_bam_dir/pass_run_IDs.txt" $bam_file
                        samtools index $bam_file
                        
                    fi    
        
                fi
            
                #################################################### STEP 6: CALL VARIANTS USING PILON ####################################################
            
            
                # the fasta file that pilon creates is the polished version. The goal of pilon is to clean the genome by removing inconsistencies, gaps, etc.
                # But it can also call variants
                # documentation: https://github.com/broadinstitute/pilon/wiki
                # --variant flag will create a VCF file. Otherwise, only a FASTA file is created
                # If you specify the --outdir flag, it will create output FASTA and VCF files with the sample_ID prefix in the directory specified after outdir
                # Xmx18g = allocate 18 GB
                if [ ! -f "$sample_pilon_dir/${sample_ID}_full.vcf.gz" ]; then
    
                    # use minimum mapping quality of 1 to exclude reads that map in multiple places (those have a mapping quality of 0)
                    pilon -Xmx30g --minmq 1 --genome "$ref_genome" --bam "$bam_file" --output "$sample_ID" --outdir "$sample_pilon_dir" --variant
        
                    # then gzip the full VCF file and delete the unzipped version. Also delete the FASTA file because it's not needed
                    gzip -c "$sample_pilon_dir/$sample_ID.vcf" > "$sample_pilon_dir/${sample_ID}_full.vcf.gz"
                    
                    rm "$sample_pilon_dir/$sample_ID.fasta"
                    rm "$sample_pilon_dir/$sample_ID.vcf"
        
                fi
        
        
                #################################################### STEP 7: CREATE OTHER FILES FOR F2_METRIC AND VARIANTS-ONLY VCF ####################################################
                
            
                # for the next two steps, check that the full VCF file exists. If not, that means all BAM files failed, so we can't proceed
                # variant-only VCF
                if [ -f "$sample_pilon_dir/${sample_ID}_full.vcf.gz" ] && [ ! -f "$sample_pilon_dir/${sample_ID}_variants.vcf" ]; then
                
                    # 1. keep all non-REF calls in another file
                    bcftools view --types snps,indels,mnps,other "$sample_pilon_dir/${sample_ID}_full.vcf.gz" > "$sample_pilon_dir/${sample_ID}_variants.vcf"
                fi
                    
                #     # 2. remove variants where FILTER contains Del, and the lengths of REF and ALT are the same. This occurs when there is a deletion upstream of this position, and this position has been deleted. 
                #         # the Del variants here typically occur because there is a mix of the reference nucleotides and the deletion result. 
                #         # -e is to exclude
                #     # 3. exclude low-mappability regions and regions with low EBR according to Marin et al., 2022
                #         # -a is the first file (A), -b is the second file (B). To pass in STDIN, use the string "stdin" according to the documentation
                #         # -v only returns entries in A that have no overlap in B
                #         # -header means to keep the header in file A before reporting the intersecting (or subtracted) results
                #     bcftools view --types snps,indels,mnps,other "$sample_pilon_dir/${sample_ID}_full.vcf.gz" | bcftools filter -e "FILTER == 'Del' & STRLEN(REF) == STRLEN(ALT)" | bedtools intersect -header -a "stdin" -b $exclude_regions_file -v > "$sample_pilon_dir/${sample_ID}_variants.vcf"
                echo "Finished variant calling for $sample_ID" # for samples that were newly processed in the current iteration
            fi
    
        fi

    else
        echo "Already finished variant calling for $sample_ID" # for samples that were already done in a previous iteration
    fi

    # for F2 calculation to determine lineage mixing: first subset the full gzipped VCF file to get only lineage-defining SNP sites
    if [ -f "$sample_pilon_dir/${sample_ID}_full.vcf.gz" ] && [ ! -f "$sample_lineage_dir/${sample_ID}_lineage_positions.vcf" ] && [ ! -f "$sample_lineage_dir/${sample_ID}_F2_Coll2014.txt" ]; then
        
        # create bcf file
        bcftools view "$sample_pilon_dir/${sample_ID}_full.vcf.gz" -O b -o "$sample_lineage_dir/${sample_ID}.bcf"

        # index bcf file
        bcftools index "$sample_lineage_dir/${sample_ID}.bcf"

        # create VCF file of just the lineage positions. Per the documentation, if --regions-file is a tab-delimited file, then it needs two columns (CHROM and POS), and POS is 1-indexed and inclusive
        # THIS IS DIFFERENT BEHAVIOR FROM IF IT WAS A BED FILE OR IF YOU USE BEDTOOLS. IN BOTH OF THOSE CASES, YOU NEED THREE COLUMNS (CHROM, BEG, AND END), AND THEY ARE 0-INDEXED WITH END BEING EXCLUSIVE (I.E. HALF-OPEN)
        bcftools view "$sample_lineage_dir/${sample_ID}.bcf" --regions-file $lineage_def_ref_pos -O v -o "$sample_lineage_dir/${sample_ID}_lineage_positions.vcf"     
    fi

    # run fast-lineage-caller using only PASS (high confidence) variants
    if [ -f "$sample_pilon_dir/${sample_ID}_variants.vcf" ] && [ ! -f "$sample_lineage_dir/${sample_ID}_lineage.tsv" ]; then
        fast-lineage-caller "$sample_pilon_dir/${sample_ID}_variants.vcf" --pass --out "$sample_lineage_dir/${sample_ID}_lineage.tsv"
    fi

done < "$input_file"


#################################### Combine SNPs on Codons so that snpEff Annotates them Correctly #####################################


# use base environment for combin_codon_variants.py and F2_calculation.py
conda deactivate

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID run_IDs_string
do

    sample_pilon_dir="$out_dir/$sample_ID/pilon"
    
    # run combine codons script to combine variants occurring on the same codon so that snpEff will properly annotate them
    if [ -f "$sample_pilon_dir/${sample_ID}_variants.vcf" ] && [ ! -f "$sample_pilon_dir/${sample_ID}_variants_combinedCodons.vcf" ]; then

        if [ ! -f "$sample_pilon_dir/${sample_ID}_variants_combinedCodons.vcf" ]; then        
            # this creates _variants_combined_codons.vcf
            python3 -u ./MtbQuantCNN/data_processing/variant_calling/combine_codon_variants.py -i "$sample_pilon_dir/${sample_ID}_variants.vcf"
        fi
        
    fi

    if [ -f "$sample_pilon_dir/${sample_ID}_variants_combinedCodons.vcf" ] && [ ! -f "$sample_pilon_dir/${sample_ID}_variants_combinedCodons.eff.vcf" ]; then
        echo "$sample_pilon_dir/${sample_ID}_variants_combinedCodons.vcf" >> $annot_fName
    fi
    
done < "$input_file"


######################################## F2 Metric: Script takes in list of sample IDs ########################################


# get the sample IDs from the input file. In the python script, the first column will be extracted
python3 -u ./MtbQuantCNN/data_processing/variant_calling/F2_calculation.py $input_file $out_dir


#################################################### snpEff Annotation ########################################################


# need this env for snpEff, bgzip, and tabix
source activate bioinformatics

# finally run snpEff if there are files there
num_files_annot=$(wc -l $annot_fName | awk '{print $1}')

if [ ! $num_files_annot -eq 0 ]; then
    
    echo "Annotating $num_files_annot files with snpEff"

    # don't add upstream and downstream gene effects, it makes the annotation rather cluttered. Include LoF annotations
    snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -no-downstream -no-upstream -lof -fileList $annot_fName

fi


# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID run_IDs_string
do

    fName="$out_dir/$sample_ID/pilon/${sample_ID}_variants_combinedCodons.eff.vcf"
    out_fName="$out_dir/$sample_ID/WHO_resistance/${sample_ID}_variants.tsv"

    if [ -f "$fName" ] && [ ! -f $out_fName ]; then
        
        if [ ! -d "$out_dir/$sample_ID/WHO_resistance" ]; then
            mkdir "$out_dir/$sample_ID/WHO_resistance"
        fi        

        # need to bgzip the VCF file to use bcftools view with the region argument. NEED TO PUT "" AROUND FILE NAME TO PROPERLY CONSIDER SPECIAL CHARACTERS IN FILENAME
        if [ ! -f "$fName.bgz" ]; then
            bgzip -c "$fName" > "$fName.bgz"
        fi
    
        # tabix the bgzipped file, which will create fName.bgz.tbi
        if [ ! -f "$fName.bgz.tbi" ]; then
            tabix -0 -p vcf "$fName.bgz" -f
        fi

        bcftools view -R ./MtbQuantCNN/data_processing/variant_calling/who_catalog_regions.bed "$fName.bgz" | SnpSift extractFields '-' POS REF ALT FILTER QUAL IMPRECISE AF DP BQ MQ IC DC ANN -e "" > $out_fName

        # remove the intermediate files for space
        rm "$fName.bgz"
        rm "$fName.bgz.tbi"
        
    fi
    
done < "$input_file"


conda deactivate

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID run_IDs_string
do

    fName="$out_dir/$sample_ID/WHO_resistance/${sample_ID}_variants.tsv"
    fName_annot="$out_dir/$sample_ID/WHO_resistance/${sample_ID}_variants_annot.tsv"
    out_fName="$out_dir/$sample_ID/WHO_resistance/${sample_ID}_pred.csv"

    if [ -f $fName ] && [ ! -f $fName_annot ]; then
        python3 -u ./MtbQuantCNN/data_processing/variant_calling/process_variants_for_who_catalog.py -i $fName
    fi
    
    if [ -f $fName_annot ] && [ ! -f $out_fName ]; then
    
        # get resistance predictions -- any Group 1 or 2 variant that passes QC leads to a prediction of R for a given drug. If not, predicted S
        python3 -u ./MtbQuantCNN/data_processing/variant_calling/who_catalog_resistance_pred.py -i $fName -o $out_fName
        python3 -u ./MtbQuantCNN/data_processing/variant_calling/who_catalog_resistance_pred.py -i $fName -o $out_fName --AF-thresh 0.25

        # get V1 predictions too, in case needed for a comparison
        python3 -u ./MtbQuantCNN/data_processing/variant_calling/who_catalog_resistance_pred.py -i $fName -o $out_fName --V1
        python3 -u ./MtbQuantCNN/data_processing/variant_calling/who_catalog_resistance_pred.py -i $fName -o $out_fName --V1 --AF-thresh 0.25
        
    fi

done < "$input_file"