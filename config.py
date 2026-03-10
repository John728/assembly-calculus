N_LIST = [16, 32]  # List of list lengths to sweep
T_STEPS = [1, 2, 4, 8, 16]   # Internal time steps for AC and UT models
N = N_LIST[-1]               # Default N for some models
K_train_max = 30             # Max hops during training
K_TEST_VALS = [1, 2, 4, 8, 10, 15, 20, 25, 30, 40, 50, 64, 80, 100]
K_TEST_VALS = sorted(list(set(K_TEST_VALS)))
NUM_CONFIGS = 2              # Number of models to test per family