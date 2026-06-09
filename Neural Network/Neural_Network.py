import numpy as np
import pickle

class Activation:
    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def relu(z):
        return np.maximum(z,0)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z,axis=-1,keepdims=True)
    
    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def sigmoid_derivative(z):
        return z*(1-z)
    
    @staticmethod
    def tanh_derivative(z):
        return 1-(z**2)
    
    @staticmethod
    def relu_derivative(z):
        return np.where(z>0,1,0)

    @staticmethod
    def softmax_derivative(s):
        if s.ndim == 1:
            return np.diag(s) - np.outer(s, s)
        return (np.eye(s.shape[1])[None,:,:] * s[:,:,None]-s[:,:,None] * s[:,None,:])

    @staticmethod
    def linear_derivative(z):
        return np.ones_like(z)

class Loss:
    @staticmethod
    def mse(aoutput,poutput):
        return np.mean((aoutput-poutput)**2)

    @staticmethod
    def mae(aoutput,poutput):
        return np.mean(np.abs(aoutput-poutput))

    @staticmethod
    def mse_derivative(aoutput,poutput):
        return 2*(poutput-aoutput)/np.size(aoutput)

    @staticmethod
    def mae_derivative(aoutput,poutput):
        return np.sign(poutput-aoutput)

class Dense:
    def __init__(self,units=1,activation="linear",input_shape=(0,2)):
        self.units=units
        self.activation=activation.lower()
        if self.activation=="relu":
            range=np.sqrt(6/input_shape[1])
        else :
            range=np.sqrt(6/(input_shape[1]+units))
        self.weights=np.random.uniform(-range,range,(units,input_shape[1]))
        self.bias=np.zeros(units)

    def forward(self,inputs):
        self.inputs = np.atleast_2d(inputs)
        # Compute z for the whole batch
        self.z = np.dot(self.inputs, self.weights.T) + self.bias
        self.a = getattr(Activation, self.activation)(self.z)
        return self.a

    def backpropagate(self,gradient,learning_rate):
        # Calculate delta for the batch
        if self.activation=="sigmoid":
            delta=Activation.sigmoid_derivative(self.a)*gradient
        elif self.activation=="tanh":
            delta=Activation.tanh_derivative(self.a)*gradient
        elif self.activation=="relu":
            delta=Activation.relu_derivative(self.z)*gradient
        elif self.activation=="softmax":
            J = Activation.softmax_derivative(self.a)
            # Matrix multiplication for batch of Jacobians
            delta = np.squeeze(J @ gradient[:, :, np.newaxis], axis=-1)
        else:
            delta=Activation.linear_derivative(self.a)*gradient

        # Pass gradient to previous layer
        delta_to_previous = np.dot(delta, self.weights)

        # Average weight and bias gradients over the batch
        batch_size = self.inputs.shape[0]
        weight_gradients = np.dot(delta.T, self.inputs) / batch_size
        bias_gradients = np.sum(delta, axis=0) / batch_size

        # Update weights and biases
        self.weights = self.weights - learning_rate * weight_gradients
        self.bias = self.bias - learning_rate * bias_gradients
        
        return delta_to_previous

class Network:
    def __init__(self,input_shape):
        self.input_shape=input_shape
        self.layers=[]
        self.network_structure=[('Input Layer   ',input_shape[1], None)]
        self.layers_count=1

    def add_layer(self,units=1,activation="linear"):
        self.layers.append(Dense(units,activation,self.input_shape))
        self.input_shape=(0,units)
        self.network_structure.append((f"Hidden Layer {self.layers_count}",units,activation))
        self.layers_count+=1
    
    def compile(self,loss="mae",learning_rate=0.01,epochs=1,batch_size=16):
        self.loss=loss.lower()
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.batch_size=batch_size

    def train(self,inputs,aoutputs):
        inputs = np.atleast_2d(inputs)
        aoutputs = np.asarray(aoutputs)
        if aoutputs.ndim == 1:
            aoutputs = aoutputs[:, np.newaxis]
        elif aoutputs.ndim == 2 and aoutputs.shape[0] == 1 and inputs.shape[0] > 1:
            aoutputs = aoutputs.T
        num_samples = len(inputs)

        for i in range(1,self.epochs+1):
            epoch_loss = 0
            indices = np.random.permutation(num_samples)
            shuffled_inputs = inputs[indices]
            shuffled_outputs = aoutputs[indices]

            for j in range(0, num_samples, self.batch_size):
                batch_x = shuffled_inputs[j:j+self.batch_size]
                batch_y = shuffled_outputs[j:j+self.batch_size]

                poutput = batch_x
                for layer in self.layers:
                    poutput = layer.forward(poutput)

                if self.loss=="mae":
                    gradient = Loss.mae_derivative(batch_y,poutput)
                    epoch_loss += Loss.mae(batch_y,poutput) * len(batch_x)
                else :
                    gradient = Loss.mse_derivative(batch_y,poutput)
                    epoch_loss += Loss.mse(batch_y,poutput) * len(batch_x)

                for layer in reversed(self.layers):
                    gradient = layer.backpropagate(gradient,self.learning_rate)

            print("Epoch : ",i,"  Loss : ",epoch_loss/num_samples)

    def print_network_structure(self):
        for i in self.network_structure :
            if i==self.network_structure[-1]:
                print("Output Layer    : ",i[1],"(",i[2],")")
                break
            print(i[0]," : ",i[1],"(",i[2],")")
    
    def predict(self, inputs):
        output = np.atleast_2d(inputs)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def save(self,path='model.pkl'):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls,path):
        with open(path, "rb") as f:
            return pickle.load(f)