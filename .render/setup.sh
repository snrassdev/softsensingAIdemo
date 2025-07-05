#!/bin/bash
# Install pyenv and set Python 3.10
curl https://pyenv.run | bash

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv install 3.10.13
pyenv global 3.10.13

python --version
