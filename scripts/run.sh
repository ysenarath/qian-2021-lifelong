cd ..
source ./venv/bin/activate
python -m src.main --train_path ./data/processed/v1/train --dev_path ./data/processed/v1/valid --test_path ./data/processed/v1/test --cand_limit 1000 --keep_all_node --fix_per_task_mem_size