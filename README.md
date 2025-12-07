# ECE1508-2025F-RL-for-Portfolio-Management

## Group Members
Ruizhe Tang: ruizhe.tang@mail.utoronto.ca

Yihang Lin: Yihang.lin@mail.utoronto.ca

Yuhan Zeng: yuhan.zeng@mail.utoronto.ca

Litao (John) Zhou: litao.zhou@mail.utoronto.ca

## Project Structure

    RL-PORTFOLIO-MANAGEMENT/
    ├─ data/ 
    │  ├─ test
    │  ├─  ├─ 30min
    │  ├─  ├─ daily
    │  ├─  └─ hourly
    │  ├─ train
    │  ├─  ├─ 30min
    │  ├─  ├─ daily
    │  ├─  └─ hourly
    │  └─ data.ipynb
    ├─ notebooks/
    │  ├─ .gitkeep
    ├─ results/
    │  ├─  ├─ figures
    │  └─  └─ metrics
    ├─ src/
    │  ├─ agents
    │  ├─  ├─ dqn_agent.py
    │  ├─  ├─ ppo_agent.py
    │  ├─  └─ ppo_lstm_agent.py
    │  ├─ baseline.py
    │  ├─ data_loader.py
    │  ├─ envs.py
    │  ├─ evaluate.py
    │  └─ features.py 
    ├─ .gitignore
    ├─ demo_all.py
    ├─ demo.py
    ├─ LICENSE
    ├─ requirement.txt
    └─ README.md

## Set up Instruction
1. Go to repository home directory and install pacakges using the following command:
        
        pip install -r requirements.txt

2. Run demo_all.py

        py demo_all.py

    or

        python3 demo_all.py

3. results will be saved in ./results and its subfolders
