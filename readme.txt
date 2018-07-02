
Chopsticks.js:

#This is made by Aden Power, and will be hereby known as "chopsticks.js". It is a neural network. Construction commenced on 18/5/18.

#Network needs to be initialised with the initNet() function. Pass in a network configuration.

#Networks configurations are in an array where the number of values is the number of layers (including input/output) and the value is the number of neurons on that layer.

#Inputs to a neural net should be passed into the "net fire" function as an array.

#
                                            
#Current cost function is quadratic error function so sum of squared error

#When you call the train function pass in one array of arrays of inputs

#The netfire function can return one of two things, if you're planning to do backprop you need to set the second argument to true and it'll return all the inputs. If you want to actually use the data for a problem then you just leave the second argument out and it will return just the outputs of the last array. If you're planning to do some online learning and need it do both put the argument to true and then do sigmoid on the last layer.



#constructTensor() creates an empty multi-dimensional array:
                                                            #First pass in an array containing the number of indexs in each dimension "shape", or, in each index of the array you can put a function which, given the "position" argument, returns that length.
                                                            #(optional) The pass in a function which will define what is in the array to start with. This function has one argument, "position" which contains an array showing where it sits in the multi dimensional array.
#The forAll() funciton is super cool: it takes one of my multi-dimensional arrays and does a function to each value. Arguments:
                                                                                        #first is the function, two arguments for that which are the position and then the current value
                                                                                        #Second is a true/false on whether to reverse the main array. This is kind of a very case specific use.