for split in train val testA testB; do
	export split
	echo ${split}
	sbatch prepare_coco_batch.sbatch
done
