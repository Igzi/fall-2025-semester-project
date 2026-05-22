git pull
huggingface-cli login
python lorauter_eval.py     --data_path dataset/combined_test.json     --res_path results/results_llama3.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type arrow
python3 lorauter_eval.py     --data_path dataset/combined_test.json     --res_path results/results_llama3.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type arrow
python3 lorauter_eval.py     --data_path dataset/combined_test.json     --res_path results_llama3/arrow.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type arrow
bash scripts/run_arrow_eval.sh 
bash scripts/run_arrow_eval_ood.sh 
bash scripts/run_spectr.sh 
bash scripts/run_spectr_ood.sh 
bash scripts/run_arrow_eval.sh 
bash scripts/run_arrow_eval_ood.sh 
bash scripts/run_spectr.sh 
bash scripts/run_spectr_ood.sh 
nvidia-smi
bash scripts/run_arrow_eval.sh 
bash scripts/run_arrow_eval_ood.sh 
bash scripts/run_spectr.sh 
bash scripts/run_spectr_ood.sh 
python3 lorauter_eval.py     --data_path dataset/combined_test.json     --res_path results/results_llama3.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type spectr
python3 summarize_results.py 
python3 summarize_results.py 
nvidia-smi
python3 ./adapter_evaluation/generate_results.py 
pip install seaborn
python3 summarize_results.py 
python3 ./adapter_evaluation/generate_results.py 
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
pip install numpy==1.24
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
python3 summarize_results.py 
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
python3 summarize_results.py 
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
python3 summarize_results.py 
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
python3 summarize_results.py 
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test2.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
python3 summarize_results.py 
python3 lorauter_eval_llama3.py     --data_path dataset/combined_test.json     --res_path results_llama3/test2.json     --config_path config/config_large.json     --lora_num 3     --model_size 7b     --eval_type mixture     --ood True
python3 summarize_results.py 
pip install nevergrad
cd lorahub_new/
python3 download_flan.py 
pip install conda install -c nvidia cuda-toolkit -y
pip install -c nvidia cuda-toolkit -y
pip3 install -c nvidia cuda-toolkit -y
pip3 install -c nvidia cuda-toolkit
pip install -c nvidia cuda-toolkit
python3 evaluate_lorahub.py 
ps -fp 78927
python3 evaluate_lorahub.py 
du -sh
nvidia-smi
