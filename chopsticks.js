/* 

#This is made by Aden Power, and will be hereby known as "chopsticks.js". It is a neural network. Construction commenced on 18/5/18.

#Networks configurations are in an array where the number of values is the number of layers (including input/output) and the value is the number of neurons on that layer.

#Inputs to a neural net should be passed into the "net fire" function as an array.

#Input to the backprop function should be in the format that is returned by the train function; an array containing two arrays:
                                            1. Contains an array for each training data set which in turn contains two more arrays:
                                                                    a) one array per layer containing the inputs into each neuron of that layer
                                                                    b) one array per layer containing the outputs from each neuron of that layer
                                                                    
                                                                    
                                            2. Contains an array for each set of training data. That array contains ideal outputs for every neuron in the output layer.
                                            
#Current cost function is simply (ideal - actual), if I want to change this need to work out the new derivative

#When you call the train function pass in one array of arrays of inputs
*/

function network () {
    
    this.lastOutputLayer;
    
    this.lastMeanError = [];
    
    this.errorArr = [];
    
    this.layers = [];
    
    this.netFire = function (inputs) {
        if (this.initialised == true) {
            var allOutputs = [];
            var allInputs = [];
            var layerOuts = [];
            var lastLayerOuts = [];
            var layerIns = [];
            
            for (let l = 0; l < this.layers.length; l++) {
               
                //we are doing this so we have a copy of the outputs of the last layer
                lastLayerOuts = JSON.parse(JSON.stringify(layerOuts));
                layerOuts = [];
                layerIns = [];
                
                
                for (let n = 0; n < this.layers[l].neurons.length; n++) {
                

                    if (l == 0) {
                        
                        layerOuts.push(inputs[n]);
                        
                        layerIns.push(inputs[n]);
                        
                        

                    }

                    else {

                        let neuronIn = 0;
                        for (let k = 0; k < this.layers[l-1].neurons.length; k++) {
                            neuronIn += this.layers[l-1].neurons[k].weights[n] * lastLayerOuts[k];
                            
                        }
                        

                        layerIns.push(neuronIn+this.layers[l].neurons[n].bias);
                        layerOuts.push(this.layers[l].neurons[n].fire(neuronIn));
                        
                    }
                    
                    
                }
                //Update the arrays of inputs and outputs
                allInputs.push(layerIns);
                    
                allOutputs.push(layerOuts);
                
            }
            

            this.lastOutputLayer = JSON.parse(JSON.stringify(layerOuts));
            
            return [allInputs,allOutputs];
        }
        else {
            throw "Network Not initalised!";
        }
    }
    
    this.config = [];
    
    this.initialised = false;
    
    this.initNet = function (config) {
        
        this.initialised = true;
        this.config = config;
        
        
        //Create layers, in each create neurons and their bias 
        for (let l = 0; l < config.length; l++) {
            if (l > 10) {
                throw "l";
            } 
            this.layers.push(new layer(l,this));
            
            for (let n = 0; n < config[l]; n++) {
                if (n > 10) {
                    throw "n";
                }   
                //Random generate a bias between 1 and - 1
                let b = (l > 0) ? (1 - (2 * (Math.round(Math.random() * 100)) / 100)) : 0;
                if (b > 10) {
                    throw "b";
                } 
                this.layers[l].neurons.push(new neuron(b,this.layers[l],this));
            }
            
        }
        
        
        //create synapses if it's not the last layer
        for (let l = 0; l < this.layers.length; l++) { 
            if (l > 10) {
                throw "l";
            } 
            if (l != this.layers.length - 1) {
                
                for (let n = 0; n < this.layers[l].neurons.length; n++) {
                    if (n > 10) {
                        throw "n";
                    } 
                    for (let k = 0; k < this.layers[l+1].neurons.length; k ++) {
                        if (k > 10) {
                            throw "k";
                        } 
                        this.layers[l].neurons[n].weights.push(parseFloat(Math.random().toFixed(2)));
                    }
                }
            }
        }   
    }
        
    this.backProp = function (trainingResults) {
        this.errorArr = [];
        var idealOuts = trainingResults[1];
        
        var compoundedNewValues = [];
        
        
        //Fill all of them with 0s in reverse order so that the changes can be added to later in the same order that they are back-propogated through
        for (let l = 0; l < this.layers.length; l++) {
            
            compoundedNewValues[l] = [];
            for (let n = 0; n < this.layers[l].neurons.length; n++) {
                compoundedNewValues[l].push([]);
                
                for (let w = 0; w < this.layers[l].neurons[n].weights.length; w++) {
                    
                    compoundedNewValues[l][n].push(0);
                }
            }
        }
        
        
        
        //Itterate for every training example"
        for (let i = 0; i < idealOuts.length; i++) {
            
            let actualIns = trainingResults[0][i][0];
            
            
            
            let actualOuts = trainingResults[0][i][1];
            let outputLayer = actualOuts[actualOuts.length-1];
            let layerError = [];
            let layerErrorDerivatives = [];
            
            
            for (let j = 0; j < outputLayer.length; j++) {
                
                layerError.push(Math.abs(idealOuts[i][j] - outputLayer[j]));
                layerErrorDerivatives.push((idealOuts[i][j] > outputLayer[j]) ? -1 : 1);
            }
            
            
            
            //ALERT: it's possible the -1 and 1 need to be switched out in the line above
            
            
            
            //Differentiate Output layer outs with respect to their net input
            let outputDerivatives = layerSigmoid(outputLayer);
            
            let totalLayerError = sum(layerError);
            this.errorArr.push(totalLayerError);
           
            
            //The array of these values
            let newValues = [];
            
            
            //An array of the deriviatives of the layer in front with respect to each neuron in this layer
            let currentDerivatives = [multiplyArrs(outputDerivatives,layerErrorDerivatives)];
            
            
            for (let l = this.layers.length - 1; l >= 0; l--) {
                
                
                let layerArray = [];
                
                //We've already done the output layer
                if (l == this.layers.length - 1) {
                    continue;
                }
                
                
                
                
                let theseDerivatives = [];
                for (let n = 0; n < this.layers[l].neurons.length; n++) {
                    
                    let neuronArray = [];
                    
                    //Work out the derivative of the neuron in the layer in front with respect to this weight
                    for (let w = 0; w < this.layers[l].neurons[n].weights.length; w++) {
                        
                        let change = actualOuts[l][n] * currentDerivatives[w];
                        
                        
                        neuronArray.push(change);
                    }
                    
                    layerArray[n] = neuronArray;
                    
                    //Calulate the derivatives of each neuron in the next layer's net input with respect to each neuron in this layer net output. Since it's linear the derivative is just the weight of the synapse going between them.
                    theseDerivatives[n] = 0;
                    
                    for (let z = 0; z < this.layers[l+1].neurons.length; z++) {
                        
                        theseDerivatives[n] += this.layers[l].neurons[n].weights[z];
                    }
                    
                    
                    
                }
                
                newValues[l] = layerArray;
                
                //Find derivative of outputs with respect to inputs
                theseDerivatives = multiplyArrs(theseDerivatives,layerSigmoid(theseDerivatives));
                
                console.log(multiplyArrs(theseDerivatives,currentDerivatives));
                //update "current derivatives" array
                currentDerivatives = multiplyArrs(theseDerivatives,currentDerivatives);
                
                
                
            }
            
            

            
            //Add to the big compounded change
            for (let l = this.layers.length - 1; l > 0; l--) {
                if (l == 0) {
                    continue;
                }
                
                for (let n = 0; n < this.layers[l].neurons.length; n++) {
                    
                    for (let w = 0; w < this.layers[l].neurons[n].weights.length; w++) {
                        
                        compoundedNewValues[l][n][w] += newValues[l][n][w];
                    }
                }
            }
            
        }
        
        //Time to update the values!!! YAY!!! FINALLY!!!! FINNNNNNNAAAALLLLY!!!!!!!!
        
        for (let l = this.layers.length - 1; l > 0; l--) {
            
                for (let n = 0; n < this.layers[l].neurons.length; n++) {
                    for (let w = 0; w < this.layers[l].neurons[n].weights.length; w++) {
                        this.layers[l].neurons[n].weights[w] -= compoundedNewValues[l][n][w];
                    }
                }
            }
        
       //Update the error
        this.lastMeanError = mean(this.errorArr);
    }
    
    
    this.train = function (ins) {
        
        var allReturnValues = [];
        
        //Itterate through each training example
        for (let i = 0; i < ins.length; i++) {
            
            //Fire the array
            returnValues = this.netFire(ins[i]);
            allReturnValues.push(returnValues);
            
        }
        
        return allReturnValues;
    }

} 


function neuron (bias,layer,net) {
    
    this.net = net;
    this.layer = layer;
    
    this.bias = bias;
    
    this.weights = [];
    
    
    this.fire = function (inputSum) {
        
        
        inputSum += this.bias;
        
        return sig(inputSum);
        
    }
}



function layer (num, net) {
    
    this.layerNum = num;
    this.neurons = [];
}
        
function sig (x) {

    return 1 / (1 + Math.pow(Math.E,-1 * x));
    
}


        
//MAE and MSE won't work
function MSE (trainingResults) {
    //Mean squared Error  
    
    var idealOuts = trainingResults[0];
    var actualOuts = trainingResults[1];
    
    let arr = [];
    
    for (let i of idealOuts) {
        
        let num = 0;
        
        for (let k = 0; k < idealOuts[i].length; k++) {
            
            num += Math.pow(idealOuts - actualOuts,2);
        }
        
        arr.push(num);
    }
    
    return mean(arr);
}
    

        
function MAE (trainingResults) {
    //Mean absolute error
    
    var idealOuts = trainingResults[0];
    var actualOuts = trainingResults[1];
    
    let arr = [];
    
    for (let k = 0; k < idealOuts[i].length; k++) {
        
        let num = 0;
        
        for (let k = 0; k < idealOuts[i].length; k++) {
            
            num += Math.abs(idealOuts - actualOuts);
        }
        
        arr.push(num);
    }
    
    return mean(arr);
}
         
function sum (numbers) {
    //input is array
    return numbers.reduce(function(a,c) {
        return a + c;
    });
}
    
function mean (numbers) {
    //input is array
    
    return sum(numbers) / numbers.length;
}


//Finds the derivative of the sigmoid function for each thing in the array
function layerSigmoid (arr) {
    
    var output = [];
    
    for (let i = 0; i < arr.length; i++) {
        output.push(arr[i]*(1-arr[i]));
    }
                    
    return output;                
}
    
function multiplyArrs (arr1,arr2) {
    
    //If they aren't equal in length return empty array
    if (arr1.length != arr2.length) {
        return [];
    }
    
    var output = [];
    
    for (let i = 0; i < arr1.length; i++) {
        
        output.push(arr1[i] * arr2[i]);
    }
    
    return output;
}


function addArrs (arr1,arr2) {
    //If they aren't equal in length return empty array
    if (arr1.length != arr2.length) {
        return [];
    }
    
    var output = [];
    
    for (let i = 0; i < arr.length; i++) {
        
        output.push(arr1[i] + arr2[i]);
    }
    
    return output;
}

function getRandomInt(min, max) {
    
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

