ckpt=$1
merge_path=$2

out_path=$3

echo "merge lora with base model"
python merge_lora.py --lora_model $ckpt --output_dir $merge_path
echo "merge done"

if [ -d "${out_path}" ]; then
    rm -r ${out_path}
fi

mkdir -p ${out_path}

cp $merge_path/* ${out_path}
cp $merge_path/*.json ${out_path}
cp $merge_path/merges.txt ${out_path}

if [ -d "${out_path}-ct2" ]; then
    rm -r ${out_path}-ct2
fi

echo "start convert ct2"

ct2-transformers-converter --model ${out_path} \
                           --output_dir ${out_path}-ct2 \
                           --copy_files vocab.json preprocessor_config.json \
                           --quantization float16

echo "Done ct2 convert"