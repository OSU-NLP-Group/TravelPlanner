export OUTPUT_DIR=/Users/xiejian/Private/code/TravelBench/results
export SET_TYPE=train
export TMP_DIR=/Users/xiejian/Private/code/TravelBench/results/train
export SUBMISSION_DIR=/Users/xiejian/Private/code/TravelBench/results/train

cd postprocess
python parsing.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --tmp_dir $TMP_DIR

# Then these parsed plans should be stored as the real json formats.
python element_extraction.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --tmp_dir $TMP_DIR

# Finally, combine these plan files for evaluation. We also provide a evaluation example file "example_evaluation.jsonl" in the postprocess folder.
python combination.py --set_type $SET_TYPE --output_dir $OUTPUT_DIR --submission_file_dir $SUBMISSION_DIR
