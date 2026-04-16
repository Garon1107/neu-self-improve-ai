#!/bin/bash
TASK="gaia"
THREADS=1
DATA_FILE_NAME="data.json"
MODAL_BASE_URL="http://localhost:8000/v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PROJECT_DIR
LLM="vllm-Qwen/Qwen3.5-0.8B"
LABEL="Qwen3.5-0.8B"
ENABLED_TOOLS="Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="dashscope-qwen2.5-7b-instruct,dashscope-qwen2.5-7b-instruct,Default,Default"
MODEL_ENGINE="trainable,dashscope,dashscope,dashscope"
DATA_FILE="$TASK/data/$DATA_FILE_NAME"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"
mkdir -p "$LOG_DIR" "$OUT_DIR"
INDICES=($(python3 -c "import json; data=json.load(open('$DATA_FILE')); print(' '.join(str(i) for i in range(len(data))))"))
new_indices=()
for i in "${INDICES[@]}"; do
    if [ ! -f "$OUT_DIR/output_$i.json" ]; then
        new_indices+=($i)
    fi
done
indices=("${new_indices[@]}")
if [ ${#indices[@]} -eq 0 ]; then
    echo "All subtasks completed."
else
    run_task() {
        local i=$1
        uv run python solve.py --index $i --task "$TASK" --data_file "$DATA_FILE" --llm_engine_name "$LLM" --root_cache_dir "$CACHE_DIR" --output_json_dir "$OUT_DIR" --output_types direct --enabled_tools "$ENABLED_TOOLS" --tool_engine "$TOOL_ENGINE" --model_engine "$MODEL_ENGINE" --max_time 300 --max_steps 10 --temperature 0.0 --base_url "$MODAL_BASE_URL" 2>&1 | tee "$LOG_DIR/$i.log"
    }
    export -f run_task
    export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM ENABLED_TOOLS TOOL_ENGINE MODEL_ENGINE MODAL_BASE_URL
    parallel -j $THREADS run_task ::: "${indices[@]}"
fi
uv run python calculate_score_unified.py --task_name "$TASK" --data_file "$DATA_FILE" --result_dir "$OUT_DIR" --response_type "direct_output" --output_file "finalresults_direct_output.json" | tee "$OUT_DIR/finalscore_direct_output.log"
