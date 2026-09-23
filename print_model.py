import torch
from engine import ShuffleEfficientViT

model = ShuffleEfficientViT( num_classes = 6 )

print("\n--- Last Block Structure ---")
print( model.efficientvit_branch.stages[-1].blocks[-1] )

# Check if there is a global final norm
print("\n--- Branch Final Norm ---")
try:
    print( model.efficientvit_branch.norm )
    print("Found 'norm' layer!")
except AttributeError:
    print("No layer named 'norm' found.")