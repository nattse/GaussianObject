#!/bin/bash

for dir in ./data/natedata/batchrun*; do
	data="$(basename $dir)"
	bash run_data.sh -r run -p $data -s 8 -z 4 -o -X loo1
	rm -r ./output/gaussian_object/run-"$data"/ckpts
	rm -r ./output/controlnet_finetune/run-"$data"/ckpts-lora
done

