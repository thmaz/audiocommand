## audiocommand 
Repository for a "proof of concept" of a convolutional neural network for speech pattern recognition. For details see the folder "docs".

How to run:

```
$ cd audiocommand
$ curl -L -o ./mini_speech_commands.zip https://www.kaggle.com/api/v1/datasets/download/antfilatov/mini-speech-commands
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

Source dataset:
https://www.kaggle.com/datasets/antfilatov/mini-speech-commands

Link to pytorch:
https://pytorch.org/get-started/locally/

Pytorch audio tutorial:
https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

Using CNN for audio classification in pytorch:
https://medium.com/@mlg.fcu/using-python-to-classify-sounds-a-deep-learning-approach-ef00278bb6ad

Convolutional Neural Networks:
https://poloclub.github.io/cnn-explainer/
https://cs231n.github.io/convolutional-networks/
