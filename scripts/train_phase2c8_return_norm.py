"""Same audited training entry; normalization admission gate is mandatory."""
import sys
from train_phase2c8_reward_ablation import main
if __name__=='__main__':
    if '--method' not in sys.argv:sys.argv.extend(['--method','v3_return_norm'])
    main()
