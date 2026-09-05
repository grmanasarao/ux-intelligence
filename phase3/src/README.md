# Phase 3 v1

One shared DeBERTa-v3-base encoder with three supervised heads:
1. multi-label UX topic
2. severity 1–5
3. sentiment

The official UXPID test split is never used for model selection. A validation subset is taken from the supplied training split, and the official test split is used only for final evaluation.

Main commands from D:\ux_research_intelligence:
python phase3\scripts\build_uxpid_training.py
python phase3\scripts\train_multitask.py --max-steps 20
python phase3\scripts\train_multitask.py --epochs 3 --batch-size 16
python phase3\scripts\evaluate_multitask.py --model-dir phase3\models\ux_multi_task_deberta
python phase3\scripts\predict_multitask.py --model-dir phase3\models\ux_multi_task_deberta --text "I could not find where to change my settings."

For an archived evidence CSV:
python phase3\scripts\predict_csv.py --model-dir phase3\models\ux_multi_task_deberta --input-csv phase3\archive\candidate_annotations.csv --output-csv phase3\data\uxpid\predictions_candidate_annotations.csv --text-column text --id-column evidence_id
