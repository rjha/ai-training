"""
The most atomic way to train and run inference for a GPT in pure, 
dependency-free Python.This file is the complete algorithm.
Everything else is just efficiency.
@karpathy
--------------------------------------
This is the annotated version of karpathy microgpt code. We have changed few 
methods, like matrix() and linear() for teaching. 
 


"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
file_name = "/Users/rjha/code/github/ai-training/scripts/output.txt" 
docs = []

with open(file_name) as f:
    for line in f:
        if line.strip():
            docs.append(line.strip())
    
random.shuffle(docs)
print(f"num docs: {len(docs)}")

"""
Let there be a Tokenizer to translate strings to sequences of 
integers ("tokens") and back 
-----------------------------------------------------------
- ''.join(docs) is all the letters in the document. 
- set(all-the-letters-in-document) will return set {A, B, C, D ...}  
  The letters can be in any order. 
- sorted({A, B, C ...}) will return letters in alphabetical order 
- unique characters in the dataset become token id
unique_chars[0] = 'a' 
unique_chars[1] = 'b' 

For an example of BPE (Byte Pair encoding) tokenizer, see tiktoken
(https://github.com/openai/tiktoken)

"""

unique_chars = sorted(set(''.join(docs))) 
BOS = len(unique_chars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(unique_chars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

"""
Let there be Autograd to recursively apply the chain rule 
through a computation graph
---------------------------------------------
To do a deep dive: check micrograd video 
https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLDM5FXaTHP1c

x ────┐
      │   ┌───┐                 ┌───┐
      ├──►│ * ├───► x * y ─────►│ + ├───► z
y ────┤   └───┘                 └───┘
      │                           ▲
      └───────────────────────────┘

Think of operations as nodes and scalar values as edges in a DAG. 
Each (output) scalar is due to input + operation, like x and y as 
input with operation * produce [x * y]. The Value class (scalar) 
for x * y will store, x and y as child (input), grad=0 and local grads 
as y and x with respect to input x and y.
  
When backward() is called on a scalar during training, then the gradient 
for scalar is computed using local gradients of input and chain rule.
     
"""


class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


"""
matrix implementation has been changed for teaching 
The karpathy version is terse and uses lambda. 

The convention is matrix(output_features, input_features) 
during computations, we multiply with transpose of the matrix, 
Y = X W^T + B 

matrix(rows, columns) assumes that wte and wpe vectors are 
column vectors (check again)
"""

import random

def matrix(num_rows, num_cols, std_deviation=0.08):
    
    grid = []
    for _ in range(num_rows):
        row = []
        for _ in range(num_cols):
            # generate random numbers for weights
            # Wrap the weights in the Value class for gradient tracking
            row.append(Value(random.gauss(0, std_deviation)))
            
        # Add finished row to our matrix grid
        grid.append(row)
        
    return grid

def print_matrix(matrix):
 
    for row_idx, row in enumerate(matrix):
        row_strings = []
        
        # Loop through each cell in the current row
        for val in row:
            # Extract the raw numeric data from the Value object
            raw_number = val.data  
            formatted_number = f"{raw_number:8.4f}"
            row_strings.append(formatted_number)
            
        # Join the numbers in the row with a space and print
        print(f"Row {row_idx:02d}: [{' '.join(row_strings) } ]")


"""
Initialize the parameters, to store the knowledge of the model
start with vocabulary and embedding dimensions. 
embedding dimension = d_model 

Our wte lookup matrix has shape vocab_size x d_model (16) 
wte -> word token embeding. Takes a token ID and turns it into a dense vector
Each token gets a d_model dimensional column vector and there are vocab_size 
such vectors in the wte lookup table. wte[index] is  of shape [1 x d_model]

wpe -> word position embedding
context length would determine the wpe size. what is called block_size below. 
This is the maximum context length of the attention window. wpe lookup table 
shape is [context_size x d_model]
wpe[index] is of shape [1 x d_model]

wte[index] + wpe [index] is fed to the transformer. 
if sentence has 3 tokens then, we get wte['t1'], wte['t2'] and wte['t3']
and add to wpe[0], wpe[1] and wpe[2]

inside the transformer, we have attention blocks and MLP (Multi layer perceptron).
Each input vector needs 3 projections for attention, Q, K, and V.
when a token vector passes through a layer, we perform the operation using the 
transposed weight matrix. The attention weight matrix shape is m x n.

Output = Token x W^T 
shape is [1 x d_model] = [ 1 x d_model] x [n x m] 
weights attn_wq, attn_wk, attn_wv and attn_wo  have shape [d_model x d_model]
if attention block has multiple heads then d_head x no_of_heads = d_model 
The results of different heads are concatenated to produce output vector [1 x d_model]

A standard Transformer MLP block requires a minimum of 2 linear layers to function 
as intended. This design allows the model to project the data up into a higher dimension 
to process features, apply a non-linear activation function, and then project it back 
down to the original embedding size. Since computations happen with W^T (transpose), the 
MLP weight matrix shape using [output_size, input_size] convention is,
UP layer (c_fc)     ->   [4 x d_model, d_model] 
Down layer (c_proj) ->   [d_model, 4 x d_model]

lm_head (Language Model Head): Takes the final dense vector and turns it back into logits 
over vocabulary. Since vocabulary size is 27 and d_model is 16, the lm_head needs 16 inputs 
to read the final hidden state and 27 outputs to assign a probability score to every single 
possible word in the vocabulary. wte and lm_head dimensions should match.  

Number of parameters for our case,

a. wte matrix - vocab_size x d_model = 27 x 16 = 432
b. wpe matrix - context_size x d_model = 16 x 16 = 256 
c. lm_head - 432 (must match wte)
d. 4 attention weight matrix, each 16 x16 - 1024 
e. MLP UP layer - 64 x 16 = 1024 
f. MLP Down layer - 16 x 64 = 1024 

Total parameters = 432 + 256 + 1024 + 1024 + 1024 + 432 = 4192.

"""

n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head

state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")


"""
Y = X * W^T + b 
X is input data, 
W is the weight matrix stored in the layer
W^T is the transpose of weight matrix 

check - are we storing transpose or the weight matrix? 

"""

def linear(x, w):
    output = []
    
    # Iterate over each row in the weight matrix. Each row 'wo' contains 
    # the weights responsible for producing ONE output feature.
    for wo in w:
        dot_product = 0
        # get individual elements of the current weight row
        # and pair them up with the corresponding elements of the input vector 'x'.
        for wi, xi in zip(wo, x):
            # Multiply the weight by the input feature and add it to the running total
            dot_product += wi * xi
            
        # Once we finish multiplying the entire row, save the final sum
        output.append(dot_product)
        
    return output

def print_vector(vector):
    print("Vector: ", end="")
    output = []
    for val in vector:
        # Format the float to 4 decimal places for clean reading
        output.append(f"{val.data:8.4f}")
    print(f"[{ ' '.join(output) } ]")



# helper functions 
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]



"""

This method processes one token at a time (no batch mode)

(1) input embeddings 
    - get token embedding on token_id 
    - get position embedding on pos_id 
    - do [ t + p ] addition 
    - normalize before hitting the transformer blocks

(2) Transformer Layer 
    - each transformer layer is composed of 2 parts, Attention & MLP
    - generate Q, K, V projections of input vector using linear(output, input)
    - append to KV cache 
    
(3) Attention block is the exact and only place where a token at position 
    t gets to “look” at tokens in the past 0..t-1. Attention is a token 
    communication mechanism.
    
 For every attention slice, 
    - q_h is for the current token,
    - k_h and v_h are slices of kv-cache (from all past tokens)
    - attention weights is calulated, 
        * compute a dot product between q_h (current query) and k_h (all past Keys)
        * apply softmax on attention_logits to get attention_weight (probabilities)
    - attention weight is multiplied with v_h (all past values) to get x_attn_slice 
    - add x_attn_slice to x_attn 
    - run x_attn through attn_wo and add residuals 
    - pass new x to MLP layer 
 (4) MLP layer 
     - set x_residual to x 
     - Normalize input before passing to MLP_FC1
     - run x through MLP_FC1 
     - apply RELU on x 
     - run x through MLP_Fc2 
     - add x_residual 

     Unlike attention, this computation is fully local to time t. The Transformer 
     intersperses communication (Attention) with computation (MLP)
     
     Note on residual: Both the attention and MLP blocks add their output back to 
     their input (x = [a + b for ...]). This lets gradients flow directly through 
     the network and makes deeper models trainable
     
  (5) Apply lm_head on input x to generate logits
       one logit per token in the vocabulary. In our case, that’s just 27 numbers. 
       Higher logit = the model thinks that corresponding token is more likely to 
       come next
  (6) KV cache growth 
      total elements = 2 x num_layers x sequence_length x d_model 
      
  
"""

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] 
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(unique_chars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

    
