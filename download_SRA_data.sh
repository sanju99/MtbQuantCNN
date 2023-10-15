#!/bin/bash
#SBATCH -c 1
#SBATCH -t 4-23:59
#SBATCH -p medium
#SBATCH --mem=5G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

module load sratoolkit/2.10.7

# this script uses entrez-direct (a command line tool installed with conda) and the sra toolkit
# entrez-direct is used to extract SRA IDs (in the form SRR) for samples, then sra toolkit downloads FASTQs

# Check if a CSV file was passed as an argument
if [[ -z "$1" ]]; then
    echo "Please provide a text file of the Biosample Accession numbers as the first argument."
    exit 1
fi

# Check if an output directory was passed as an argument
if [[ -z "$2" ]]; then
    echo "Please provide an output directory as the second argument."
    exit 1
fi

no_SRA_ID_file="/home/sak0914/MtbQuantCNN/not_found_FASTQs.txt"
finished_file="/home/sak0914/MtbQuantCNN/found_FASTQs.txt"

# The parent directory where sample directories should be created
parent_dir="$2"

# Get line count of the CSV file
total_lines=$(wc -l < "$1")

echo "Total lines in CSV file: $total_lines"

line_count=0

# Read the CSV file
for line in $(cat "$1")
do
    line_count=$((line_count + 1))

    # Split the line into BioSample ID and sample ID
    IFS=',' read -r sample_id <<< "$line"

    echo "Processing line $line_count of $total_lines"

    # remove any - characters if they are there. Need them above to search the SRA database
    formatted_sample_id=$(echo "$sample_id" | sed 's/-//g')

    # Create a directory for the sample in the parent directory
    sample_dir="$parent_dir/$formatted_sample_id"
    
    if [ ! -f "$sample_dir/${formatted_sample_id}_R1.fastq.gz" ] || [ ! -f "$sample_dir/${formatted_sample_id}_R2.fastq.gz" ]; then

        # Fetch SRR numbers for the BioSample ID. Take the last one (highest, most recent in the case of resequences)
        srr_id=$(esearch -db sra -query $sample_id | efetch -format runinfo | grep -v "Run" | cut -d ',' -f1 | tail -1)
        
        # check if the variable is NOT empty, which means an SRR ID was found
        if [ ! -z "$srr_id" ]; then

            echo "Biosample accession: $formatted_sample_id, SRA ID: $srr_id"
    
            if [ ! -d "$sample_dir" ]; then
                mkdir "$sample_dir"
            else
                # if it exists, delete it (and the contents) and remake it
                rm -R "$sample_dir"
                mkdir "$sample_dir"
            fi
        
            fastq-dump --split-files --outdir "$sample_dir" $srr_id
    
            # FASTQ files are named with the SRR ID
            FQ1="$sample_dir/${srr_id}_1.fastq"
            FQ2="$sample_dir/${srr_id}_2.fastq"
    
            # gzip and rename with the sample ID
            gzip -c $FQ1 > "$sample_dir/${formatted_sample_id}_R1.fastq.gz"
            gzip -c $FQ2 > "$sample_dir/${formatted_sample_id}_R2.fastq.gz"
    
            rm $FQ1
            rm $FQ2

            # check that the original FASTQ files have the same numbers of lines
            FQ1_line_count=$(gunzip -c "$sample_dir/${formatted_sample_id}_R1.fastq.gz" | wc -l)
            FQ2_line_count=$(gunzip -c "$sample_dir/${formatted_sample_id}_R2.fastq.gz" | wc -l)
        
            # check that neither FASTQ file has no reads
            if [ $FQ1_line_count -eq 0 ] || [ $FQ2_line_count -eq 0 ]; then
                echo "Error: $$sample_dir/${formatted_sample_id}_R1.fastq.gz or $$sample_dir/${formatted_sample_id}_R2.fastq.gz has no reads"
                exit 1
            # Compare the counts and raise an error if they are not equal 
            elif [ "$FQ1_line_count" -ne "$FQ2_line_count" ]; then
                echo "Error: $sample_dir/${formatted_sample_id}_R1.fastq.gz and $sample_dir/${formatted_sample_id}_R2.fastq.gz have different line counts"
                exit 1
            else
                echo "Line counts in paired-end FASTQ files for $formatted_sample_id match"
            fi

            echo $sample_id $srr_id >> $finished_file
        
        else
            echo "No SRA ID found for $sample_id"
            echo $sample_id >> $no_SRA_ID_file
        fi    
    
    else
        echo "FASTQ files for $sample_id have already been downloaded and gzipped"
    fi

    echo "-------------------------------"
    
done