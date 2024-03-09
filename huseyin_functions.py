def get_run_name():
    with open("/home/huseyin/fungtion/DANN_py3/ilveilceler.txt", "r+", encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            character_name = lines[0].strip()
            file.seek(0)  # Dosyanın başına dön
            for line in lines[1:]:
                file.write(line)  # İlk satırı sil
            file.truncate()  # Dosyanın geri kalanını kes
            file.write(character_name + "\n")  # Karakteri dosyanın sonuna ekle
            return character_name
        else:
            return None

from torch.nn.modules.module import _addindent
import torch
import numpy as np

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

def distribute_apples(M, N):
  """
  Distributes M apples into N slots as balanced as possible.

  Args:
      M: The total number of apples to distribute.
      N: The number of slots.

  Returns:
      A list of length N representing the number of apples in each slot.
  """

  # Base case: If there's only one slot, put all apples there.
  if N == 1:
    return [M]

  # Calculate the ideal number of apples per slot (might have decimals).
  ideal_apples_per_slot = M / N

  # Initialize the list with the floor of the ideal distribution.
  distribution = [int(ideal_apples_per_slot)] * N

  # Distribute the remaining apples (decimals) one by one to slots with less.
  remaining_apples = M - sum(distribution)
  for _ in range(remaining_apples):
    # Find the first slot with less than the ideal amount.
    for i in range(N):
      if distribution[i] < ideal_apples_per_slot:
        distribution[i] += 1
        break
    
  return distribution