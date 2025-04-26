from pathlib import Path
import pandas as pd


current_dir = Path(__file__).parent
LORA_LONGLIST = dict()

for v in ['sd15', 'sdxl']:

    # Load LoRA details
    main_df = pd.read_csv(str(current_dir / f"./{v}_lora.csv"))
    main_df = main_df.dropna(subset=['id','category','article','download_url'])

    # Load LoRA examples to display
    demo_df = pd.read_csv(str(current_dir / f"./{v}_lora_demo.csv"))
    demo_df = demo_df.groupby(['id'])['demo_url'].apply(list).reset_index(name='demo_url')

    # Combine
    LORA_LONGLIST[v] = main_df.merge(demo_df, how='left', on=['id']).dropna(subset=['id'])

