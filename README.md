# VoiceClone
## Dependencies

   This code was tested on Python 3.8 with PyTorch 1.10.0. 
   [Poetry]('https://python-poetry.org/') was chosen for package management.
   Packages can be installed by:

   ```bash
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

   poetry install
   poetry shell
   ```

## Dataset
    
   For this task was chosen dataset VCTK.
   Because it contains the same phrases spoken by different speakers.
   Working with it takes place through the wrapper of a torchaudio
   For preprocessing data you must execute next commands:
   ```bash
   python preprocessing.py # it will take a long time 
   ```

## Train 

   For start train execute next command
   ```bash
   python train.py 
   ```
   Logs can be viewed in the interface tensorboard
   ```bash
   tensorboard --logdir logs
   ```

## Author

   Gaivoronsky Alexander lifami40@gmail.com