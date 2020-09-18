ORIGINALPATH=$(pwd)

CURRENT_PATH_IMAGE="${ORIGINALPATH}/DUTS-TR/DUTS-TR-Image"
CURRENT_PATH_MASK="${ORIGINALPATH}/DUTS-TR/DUTS-TR-Mask"
CURRENT_PATH_DST="${ORIGINALPATH}/DUTS-TR/DUTS-TR-Inpainted"

mkdir ${CURRENT_PATH_DST}

rm ./paths_input_mask_output.txt
rm ./paths_input_mask.txt

for i in `ls -v ${CURRENT_PATH_IMAGE}/*.jpg`;
do

# basename "$i"
f="$(basename -- $i)"
filename="${f%.*}"

echo ${CURRENT_PATH_IMAGE}/"$f" ${CURRENT_PATH_MASK}/"$filename".png ${CURRENT_PATH_DST}/"${filename}"_inpaint.jpg >> paths_input_mask_output.txt
echo ${CURRENT_PATH_IMAGE}/"$f" ${CURRENT_PATH_MASK}/"$filename".png >> paths_input_mask.txt

done
