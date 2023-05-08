TRUST_df=$1
out_dir=$2
# awk -F',' '{print $NF}' $TRUST_df

read header

while IFS=',' read -r sample_id vcf_file
do
    # do something with the last column value
    #echo "Sample ID: $sample_id"

    # do other steps with the values in the other columns
    # ...
    bcftools view --types snps,indels,mnps,other $vcf_file > "$out_dir/$sample_id.vcf"
    #echo "$out_dir/$sample_id.vcf"

done < $TRUST_df