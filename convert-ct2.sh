ckpt=$1
base=$2
out_path=$3

if [ -d "${out_path}" ]; then
    rm -r ${out_path}
fi

mkdir -p ${out_path}

cp $base/*.json ${out_path}
cp $base/merges.txt ${out_path}
cp $ckpt/* ${out_path}

if [ -d "${out_path}-ct2" ]; then
    rm -r ${out_path}-ct2
fi

echo "start convert ct2"

ct2-transformers-converter --model ${out_path} \
                           --output_dir ${out_path}-ct2 \
                           --copy_files vocab.json preprocessor_config.json \
                          --quantization float16

echo "Done ct2 convert"