"""Continue bounded V2 optimization only after development and admission gates."""
from train_phase2c6_reward_ablation import main

if __name__=="__main__":
    main(curriculum=True)
