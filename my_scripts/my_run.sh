python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --model gpt-3.5-turbo \
  --result_dir result


python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 27 \
  --test_end_idx 32 \
  --model gpt-4o \
  --result_dir result27_32

sudo apt-get install libatk1.0-0\                
    libatk-bridge2.0-0\                          
    libcups2\                                    
    libatspi2.0-0\                               
    libxcomposite1\                              
    libxdamage1\                                 
    libpango-1.0-0\                              
    libasound2      



python my_run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 27 \
  --test_end_idx 28 \
  --model gpt-4o \
  --result_dir result_my_run \
  --max_steps 10