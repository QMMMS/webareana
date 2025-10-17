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



python stage_1_explore_code.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_explore.json \
  --test_start_idx 812 \
  --test_end_idx 813 \
  --model gpt-4o \
  --result_dir result_stage_1_explore \
  --max_steps 10