#install Falcontune
#git clone https://github.com/rmihaylov/falcontune

#install its dependencies
#cd falcontune
#pip install -r requirements.txt 
#python setup.py install


#get Falcon-7B model
#wget https://huggingface.co/TheBloke/falcon-7b-instruct-GPTQ/resolve/main/gptq_model-4bit-64g.safetensors

#toy dataset
#wget https://github.com/gururise/AlpacaDataCleaned/raw/main/alpaca_data_cleaned.json

#Fine-tune Falcon-7B on toy dataset
#24h on a free Google Colab GPU
falcontune finetune \
    --model=falcon-7b-instruct-4bit \
    --weights=./gptq_model-4bit-64g.safetensors \
    --dataset=./alpaca_data_cleaned.json \ #choose your own dataset
    --data_type=alpaca \
    --lora_out_dir=./falcon-7b-instruct-4bit-alpaca/ \
    --mbatch_size=1 \
    --batch_size=2 \
    --epochs=3 \
    --lr=3e-4 \
    --cutoff_len=256 \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --warmup_steps=5 \
    --save_steps=50 \
    --save_total_limit=3 \
    --logging_steps=5 \
    --target_modules='["query_key_value"]' \
    --backend=triton