# merge-lora
CUDA_VISIBLE_DEVICES=0 verl export \
    --adapters swift/test_bert \
    --output_dir output/swift_test_bert_merged \
    --merge_lora true

# bnb quantize
CUDA_VISIBLE_DEVICES=0 verl export \
    --model output/swift_test_bert_merged \
    --output_dir output/swift_test_bert_bnb_int4 \
    --quant_bits 4 \
    --quant_method bnb

# infer
CUDA_VISIBLE_DEVICES=0 verl infer \
    --model output/swift_test_bert_bnb_int4
