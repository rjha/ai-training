

# How to activate python vm 

$cd ~/sw/pyvm/ai-training 
$source ./bin/activate 

# How to start Jupypter notebook
# Jupyter binaries are part of python vm 
# @imp This command should be run from an activated vm prompt 
# (ai-training) mbair... 
$jupyter notebook 

# if terminal does not find Jupyter then add path
export PATH=$PATH:~/sw/pyvm/ai-training/bin 


