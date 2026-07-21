
# machinery from micro gpt example
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos
import pandas as pd
from dataclasses import dataclass, field
import json



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


def calculate_bce_loss(logit, target_label):
    """
    Computes Binary Cross-Entropy loss while strictly maintaining 
    the custom autograd graph structure.
    """
    # 1. Compute prob = 1 / (1 + exp(-logit))
    # By starting with 1 / (Value), we ensure the whole chain remains a Value node
    prob = 1 / (1 + (-logit).exp())
    
    # 2. Extract loss depending on binary ground truth target
    if target_label == 1:
        # If target is 1, loss = -log(prob)
        loss = -prob.log()
    else:
        # If target is 0, loss = -log(1 - prob)
        # Use a Value node for the number 1 to guarantee __sub__ operations track history
        one_node = Value(1.0)
        loss = -(one_node - prob).log()
        
    return loss


def scale_vector(vector):
    """
    Standardises a single 1D vector (Z-score normalization).
    Returns a list of scaled values.
    """
    mean = sum(vector) / len(vector)
    # Calculate variance to get the standard deviation
    variance = sum((x - mean) ** 2 for x in vector) / len(vector)
    std = variance ** 0.5
    
    # Avoid division by zero if a column has zero variance
    if std == 0:
        return [0.0 for x in vector]
        
    return [(x - mean) / std for x in vector]



class DataLoader:
    def __init__(self, data_path: str, sep: str = ','):
        self.data_path = data_path
        self.sep = sep
        
        # Internal cache fields for data matrices
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
    
    def process(self, train_split: float = 0.8, seed: int = 42):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"error: file not found at '{self.data_path}'")
        
        try:
            df = pd.read_csv(self.data_path, sep=self.sep)
            print(df.head())
        except Exception as e:
            raise IOError(f"error reading file: {e}")

        # 2. Structural Column Verification
        required_cols = ['alcohol', 'volatile acidity', 'quality']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Schema violation: Input data file must contain columns: {required_cols}")

        # 3. Vector Feature Engineering & Scaling
        scaled_alcohol = scale_vector(df['alcohol'].tolist())
        scaled_acidity = scale_vector(df['volatile acidity'].tolist())
        
        # 4. Binary Output Map Configuration
        X_all = [list(features) for features in zip(scaled_alcohol, scaled_acidity)]
        y_all = (df['quality'] >= 6).astype(int).tolist()

        # 5. Stratified Partitioning & Shuffling
        self.data_pairs = list(zip(X_all, y_all))
        random.shuffle(self.data_pairs)
        
        split_idx = int(len(self.data_pairs) * train_split)
        train_set = self.data_pairs[:split_idx]
        test_set = self.data_pairs[split_idx:]
        
        # Unzip clean structural pairs back to instance level properties
        self._X_train, self._y_train = zip(*train_set) if train_set else ([], [])
        self._X_test, self._y_test = zip(*test_set) if test_set else ([], [])
        
        # Convert internal tuples back into structural standard lists
        self._X_train, self._y_train = list(self._X_train), list(self._y_train)
        self._X_test, self._y_test = list(self._X_test), list(self._y_test)

    def debug(self):
        print(f"Total dataset size: {len(self.data_pairs)}")
        print(f"Training set size:  {len(self._X_train)} (80%)")
        print(f"Testing set size:   {len(self._X_test)} (20%)")

    def get_train_data(self) -> tuple:
        """Returns the training feature matrix and ground truth labels (X, y)."""
        if self._X_train is None:
            raise RuntimeError("Data state error: Invoke load_and_process() before extracting records.")
        return self._X_train, self._y_train

    def get_test_data(self) -> tuple:
        """Returns the test validation feature matrix and ground truth labels (X, y)."""
        if self._X_test is None:
            raise RuntimeError("Data state error: Invoke load_and_process() before extracting records.")
        return self._X_test, self._y_test



@dataclass
class ModelParameters:
    input_dim: int = 2
    hidden_dim: int = 4
    output_dim: int = 1
    
    # Structural layer containers
    w_hidden: list = field(default_factory=list, init=False)
    b_hidden: list = field(default_factory=list, init=False)
    w_output: list = field(default_factory=list, init=False)
    b_output: list = field(default_factory=list, init=False)

    def __post_init__(self):
        """Uses your custom matrix helper to build layers instantly."""
        # Hidden Layer (4 neurons x 2 inputs) + 4 biases
        self.w_hidden = matrix(self.hidden_dim, self.input_dim)
        # self.b_hidden = [Value(0.0) for _ in range(self.hidden_dim)]
        self.b_hidden = [Value(0.01) for _ in range(self.hidden_dim)]

        # Output Layer (1 neuron x 4 inputs) + 1 bias
        self.w_output = matrix(self.output_dim, self.hidden_dim)
        self.b_output = [Value(0.0) for _ in range(self.output_dim)]

    def get_flat_params(self) -> list:
        """Flattens all 17 active Value tracking nodes for the optimizer loop."""
        flat_w_hidden = [w for neuron in self.w_hidden for w in neuron]
        flat_w_output = [w for neuron in self.w_output for w in neuron]
        return flat_w_hidden + self.b_hidden + flat_w_output + self.b_output


class DataTrainer:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 150):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self, x_sample: list, params: ModelParameters) -> Value:
        """
        Executes a shallow, graph-safe forward pass for a single data row.
        """
        # Wrap raw inputs to ensure standard Value-to-Value graph tracking
        x_nodes = [Value(val) for val in x_sample]
        
        # 1. Hidden Layer: (W * x) + b -> ReLU
        hidden_linear = linear(x_nodes, params.w_hidden)
        hidden_raw = [h_val + b_val for h_val, b_val in zip(hidden_linear, params.b_hidden)]
        hidden_activated = [h_node.relu() for h_node in hidden_raw]
        
        # 2. Output Layer: (W * hidden)
        output_linear = linear(hidden_activated, params.w_output)
        
        # Extract the scalar logit out of the 1-element list and add the scalar bias
        final_logit = output_linear[0] + params.b_output[0]
        
        return final_logit

    def train(self, data_loader: DataLoader, params: ModelParameters):
        """
        Recursion-safe batch optimization loop. Accumulates gradients 
        row-by-row and updates weights once per epoch to prevent oscillation.
        """
        X_train, y_train = data_loader.get_train_data()
        
        # Initialize Adam state vectors based on the initial parameter length
        flat_params_init = params.get_flat_params()
        m = [0.0] * len(flat_params_init)
        v = [0.0] * len(flat_params_init)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t = 0 

        print(f"Beginning Optimization Engine across {self.epochs} epochs...")

        for epoch in range(self.epochs):
            epoch_loss_float = 0.0
            
            # Create a dictionary to accumulate gradients safely across rows
            # This prevents building a deep, recursive graph structure
            accumulated_grads = {id(p): 0.0 for p in params.get_flat_params()}
            
            # 1. Accumulate Gradients Sample-by-Sample
            for x_sample, y_target in zip(X_train, y_train):
                # Forward pass creates an isolated, tiny graph for just 1 row
                logit = self.forward(x_sample, params)
                sample_loss = calculate_bce_loss(logit, y_target)
                
                epoch_loss_float += sample_loss.data
                
                # Clear gradients for this individual sample's isolated graph
                active_params = params.get_flat_params()
                for p in active_params:
                    p.grad = 0.0
                    
                # Backward pass calculates gradients for just this row (Depth ~5 nodes)
                sample_loss.backward()
                
                # Accumulate the numerical gradients into our tracking dictionary
                for p in active_params:
                    accumulated_grads[id(p)] += p.grad
            
            # 2. Epoch-Level Parameter Optimization Step (True Batch Update)
            t += 1
            active_params = params.get_flat_params()
            N = len(X_train)
            
            for i, p in enumerate(active_params):
                # Calculate the true average gradient across the entire dataset
                avg_grad = accumulated_grads[id(p)] / N
                
                # Run the standard Adam optimizer update using the stabilized average gradient
                m[i] = beta1 * m[i] + (1 - beta1) * avg_grad
                v[i] = beta2 * v[i] + (1 - beta2) * (avg_grad ** 2)
                
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                
                # Mutate the parameter data float directly
                p.data -= self.learning_rate * m_hat / (math.sqrt(v_hat) + eps)
            
            # Print average training summary metrics
            if (epoch + 1) % 10 == 0 or epoch == 0:
                average_bce = epoch_loss_float / N
                print(f"Epoch {epoch+1:3d} | Average Training BCE Loss: {average_bce:.5f}")

    def save_weights(self, params: ModelParameters, filepath: str = "wine_weights.json"):
        """
        Serializes the final trained 17 scalar floating points cleanly into disk storage.
        """
        weights_and_biases = {
            "w_hidden": [[p.data for p in row] for row in params.w_hidden],
            "b_hidden": [p.data for p in params.b_hidden],
            "w_output": [[p.data for p in row] for row in params.w_output],
            "b_output": [p.data for p in params.b_output]
        }
        
        with open(filepath, "w") as f:
            json.dump(weights_and_biases, f, indent=4)
            
        print(f"Weights successfully archived to persistent file: '{filepath}'")


    def evaluate(self, data_loader: DataLoader, params: ModelParameters):
        """
        Evaluates the model's classification accuracy on the 20% test data partition.
        """
        # 1. Fetch the testing dataset
        X_test, y_test = data_loader.get_test_data()
        
        if not X_test:
            print("Test data is empty. Please verify your data split or partitioning.")
            return

        correct_predictions = 0
        total_samples = len(X_test)
        
        print("\n" + "="*40)
        print("Starting Model Evaluation on 20% Test Partition...")
        print("="*40)

        # 2. Iterate through the test dataset
        for x_sample, y_target in zip(X_test, y_test):
            # Run the forward pass with frozen weights (we don't calculate gradients here)
            logit = self.forward(x_sample, params)
            
            # Apply Sigmoid to get probability (between 0 and 1)
            prob = 1 / (1 + (-logit).exp())
            
            # Make a binary prediction (Threshold = 0.5)
            prediction = 1 if prob.data >= 0.5 else 0
            
            # 3. Check if the prediction matches the target label
            if prediction == y_target:
                correct_predictions += 1
                
        # 4. Compute and display classification accuracy
        accuracy = (correct_predictions / total_samples) * 100.0
        
        print(f"Evaluation Complete!")
        print(f"Correct Predictions: {correct_predictions} out of {total_samples}")
        print(f"Test Accuracy:       {accuracy:.2f}%")
        print("="*40)
        
        return accuracy


def main():
    print("main")
    file_name = '/Users/rjha/code/github/ai-training/data/uci/wine/red.csv'
    data_loader = DataLoader(file_name, sep = ";")
    data_loader.process()
    data_loader.debug() 

    params = ModelParameters(input_dim=2, hidden_dim=4)
    print(f"Initial parameters count: {len(params.get_flat_params())}")
    
    trainer = DataTrainer(learning_rate=0.05, epochs=500)
    trainer.train(data_loader, params) 
    trainer.save_weights(params)
    trainer.evaluate(data_loader, params)
    

if __name__ == "__main__":
    main()