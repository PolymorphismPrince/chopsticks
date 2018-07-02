


Array.prototype.knownDimensions = 1;
Array.prototype.multiDimensional = false;
function constructTensor (shape,method,position) {
	method = method || function () {return 0;}
    position = position || [];
  	
   if (shape.length == 0) {return method(position);}

	var output = [];
	output.knownDimensions = shape.length;
    if (shape.length > 1) {output.multiDimensional = true;}
  
	for (let i = 0; i < shape[0]; i++) {
  	let positionTemp = position.slice();
    positionTemp.push(i);
    var shapeTemp = shape.slice(1);
    
    if (typeof shapeTemp[0] == "function") {shapeTemp[0] = shape[1](positionTemp);}
    
  	output[i] = constructTensor(shapeTemp,method,positionTemp);
    
  }
  return output;
}




Array.prototype.forAll = function (func,reverse,position) {
	position = position || [];
	reverse = reverse || false;  
  
  
  
	if (reverse) {
			for (let i = this.length - 1; i >= 0; i--)  {
      	
      	let positionTemp = position.slice();
        positionTemp.push(i);
        
        this.forAll.apply(this[i],[func,false,positionTemp]);
        
      }
  }
  else {
  
  	for (let i = 0; i < this.length; i++)  {
    		
      	let positionTemp = position.slice();
        positionTemp.push(i);ind
        
        if (this.multiDimensional == false) {
    
          this[i] = func(positionTemp,this[i]);
          
          continue;
      	}
        
        this.forAll.apply(this[i],[func,false,positionTemp]);
        
      }
  }
}

var initialisedNetworks = [];

//Event listener for key short cuts
document.addEventListener("keydown", function(e) {
    if (e.keyCode == 71) {
        GUI.toggle();
    }
}, false);


var placeholder;

//GUI
GUI = {
    
    toggled : false,
    
    graph : function () {
        
    },
    
    update : function () {
        
    },
    
    toggle : function() {
          
        if (this.toggled == true) {
            
            
            document.body.removeChild(this.GUIDiv);
            this.toggled = false;
            clearInterval(this.loop);
        }
        
        else {
            this.toggled = true;
            document.body.innerHTML += "<div id='netGUI' style='width:100%; height:100%; background-color: grey; position: absolute; top: 0px; left: 0px; opacity: 0.5;'> </div> ";
            
            this.GUIDiv = document.getElementById("netGUI");
            
            placeholder = this.GUIDiv;
            this.GUIDiv.innerHTML += "<button onclick='GUI.toggle()'> Close </button>";
            this.loop = setInterval(this.update,500);
        }
    }
}

function network () {
    
    this.errorHistory = [];
    
    this.fires = 0;
    
    this.lastOutputLayer;
    
    this.lastMeanError = [];
    
    this.errorArr = [];
    
    this.layers = [];
    
    this.netFire = function (inputs,backProp) {
        
        this.fires++;
        
        if (this.initialised == true) {
            
            var lastOutputs = inputs.slice();
            var config = this.config;
            var allInputs = constructTensor([this.config.length + 1,function (pos) {
                
                return config[pos[0]];
            }]);
            allInputs[0] = inputs;
            
            
            
            for (let l = 0; l < this.config.length; l++) {
                let lastOutputsTemp = sig(lastOutputs);
                
                
                for (let n = 0; n < this.config[l]; n++) {
                    
                    
                    
                    lastOutputs[n] = sum(multiplyArrays(lastOutputsTemp,this.weights[l][n])) + this.biasses[l][n];
                    
                    allInputs[l + 1][n] = lastOutputs[n];
                    
                } 
            
            }
            
            
        }
        else {
            throw "network error: Network not initalised!";
        }
        
        return (backProp == true) ? allInputs : sig(allInputs[allInputs.length-1]);
        
    }
    
    
    this.initialised = false;
    
    this.initNet = function (config) {
        
        if (this.initialised) {
            throw "network error: Network already initalised!";
        }
        this.initialised = true;
        this.inputSize = config[0];
        this.config = config.slice(1);
        
        
        this.weights = constructTensor([config.length-1,function (pos) {
            return config[pos[0] + 1];
        },function (pos) {
            return config[pos[0]];
        }],function (pos) {
            return parseFloat(((Math.random().toFixed(3)) * 2 - 1).toFixed(2));
        });
        
        
        
        this.biasses = constructTensor([config.length - 1, function (pos) {
            return config[pos[0] + 1];
        }],function (pos) {
            return parseFloat(((Math.random().toFixed(3)) * 2 - 1).toFixed(2));
        });
        
        initialisedNetworks.push(this);
        
    }

    //Default learning rate:
    this.learningRate = 0.01;
        
    this.backProp = function (inputs,desiredOutputs,error) {
        
        var newWeights = JSON.parse(JSON.stringify(this.weights));
        var newBiasses = JSON.parse(JSON.stringify(this.biasses));
        
        
        //Itterate for every training example"
        for (let i = 0; i < desiredOutputs.length; i++) {
            
            let specificInputs = inputs[i];
            let specificDesiredOutputs = desiredOutputs[i];
            
            let derivatives = [];
            let outputLayer = JSON.parse(JSON.stringify(specificInputs[specificInputs.length - 1]));
            
            outputLayer.forEach (function (element,index) {
                
                //Differentiate the total error with respect to each output: the function is f(x) = (y - x)^2 + z. The derivative, therefore, is f(x) = 2y - 2x.
                derivatives.push(specificDesiredOutputs[index] * 2 - sig(element) * 2);
            });
            
            

            //Itterate through each layer backwards and do the weights and biasses
            for (let l = this.config.length - 1; l >= 0; l--) {
                
                //Differentiate Output layer outs with respect to their net input. function is sigmoid, derivative of sigmoid is sig(x) * (1 - sig(x))
                derivatives = multiplyArrays(layerSigmoidDerivative(inputs[i][l + 1]),derivatives);
                
                //Make the derivatives really small and the reverse of their sign for the change so we don't overshoot.
                let changes = multiplyArrays(derivatives.slice().fill(this.learningRate * 1),derivatives);
                
                
                //Differentiate biasses:
                //Biasses are added, this means the derivatives are 1 and so using the chain rule it's just the derivative of net input of the layer. (1x = x obviously)
                newBiasses[l] = addArrays(changes,newBiasses[l]);
                
                let derivativesTemp = [];
                //Because foreach uses an anoynmous function, "this" is not the network
                let currentNetwork = this;
                
                //Differntiate weights and net outputs of the next layer backwards.
                this.weights[l].forEach(function (neuron,neuronNum) {
                    
                    //This is completly linear here: y = wn + b. (b is the rest of the weighted sum, w is the current weight and n is the output of the last layer. The gradient is just the output of the last layer. and for the derivative of the output of the last layer, you do the opposite. The gradient is just the weight. )
                    
                    currentNetwork.weights[l][neuronNum].forEach(function(weight,weightNum) {
                        
                        
                        //Update the newWeights array
                        newWeights[l][neuronNum][weightNum] += sig(specificInputs[l][weightNum]) * changes[neuronNum];
                        
                        
                        
                        //Update a temporary value of the derivatives, we add to it instead of just updating it because it's the sum of the derivatives of all the neurons in the next layer with respect to it
                        derivativesTemp[weightNum] = derivativesTemp[weightNum] || 0;
                        derivativesTemp[weightNum] += weight * derivatives[neuronNum];
                    });
                    
       
                });
                
                derivatives = derivativesTemp.slice();
                 
            }
            
        }
            
        //Update the actual weights and biasses:
        this.weights = JSON.parse(JSON.stringify(newWeights));
        this.biasses = JSON.parse(JSON.stringify(newBiasses)); 
    }
    
    
    this.train = function (ins,expectedOuts) {
        
        var allReturnValues = [];
        var allError = [];
        var totalError;
        //Itterate through each training example
        for (let i = 0; i < ins.length; i++) {
            
            //Fire the array
            returnValues = this.netFire(ins[i],true);
            
            allReturnValues.push(returnValues);
            if (expectedOuts != undefined) {
                allError.push(sumSquaredError(returnValues[returnValues.length - 1],expectedOuts[i]));
                
            }
            

        }
        
        if (expectedOuts != undefined) {
            
            totalError = mean(allError);
            this.backProp(allReturnValues,expectedOuts,allError);
            return totalError;
        }
        else {
        
            return allReturnValues;
        }
    }

} 


function sumSquaredError(arr1,arr2) {
    var difference = subtractArrays(arr2,arr1);
    
    for (let i = 0; i < difference.length; i++) {
        difference[i] = Math.pow(difference[i],2);
        
    }
    
    return sum(difference);
    
}



        
function sig (x) {
    
    if (typeof x == "object") {
        let output = [];
        x.forEach (function (element) {
            
            output.push(1 / (1 + Math.pow(Math.E,-1 * element)));
            
        });
        return output;
    }
    return 1 / (1 + Math.pow(Math.E,-1 * x));
    
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
function layerSigmoidDerivative (arr) {
    
    var output = [];
    
    
    for (let i = 0; i < arr.length; i++) {
        
        output.push(sig(arr[i])*(1-sig(arr[i])));
        
    }
    
                    
    return output;                
}
    
function multiplyArrays (arr1,arr2) {
    
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

function addArrays (arr1,arr2) {
    
    //If they aren't equal in length return empty array
    if (arr1.length != arr2.length) {
        return [];
    }
    
    var output = [];
    
    for (let i = 0; i < arr1.length; i++) {
        
        output.push(arr1[i] + arr2[i]);
    }
    
    return output;
}

function subtractArrays (arr1,arr2) {
    
    //If they aren't equal in length return empty array
    if (arr1.length != arr2.length) {
        return [];
    }
    
    var output = [];
    
    for (let i = 0; i < arr1.length; i++) {
        
        output.push(arr1[i] - arr2[i]);
    }
    
    return output;
}

function getRandomInt (min,max) {
    return Math.floor(Math.random() * (max - min) + min);
}




