lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
for lg_pair in $lg_pairs; do
  ./get-data-para.sh $lg_pair
done
