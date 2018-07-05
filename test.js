var trainingSetSize = 300;
var currentDiv = "";
var intervalSize = 1000;
function rule (x,y) {
    
    if ((Math.pow(x,5) + y) > 1 || (Math.pow(x,5) + y) < 0.5) {
        return 0;
    }
    
    return 1;
}




var test = new network();


//If the first neuron is big that means we are going red
//Else we are going blue

function judge (x,y,x1,x2) {
    var testResults = test.netFire([x * 2 -1,y * 2 -1,x1 * 2 -1,x2 * 2 -1]);
    
    
    return (testResults[0] > testResults[1]) ? 1 : 0;
}



test.initNet([4,1,2]);

test.biasses = [[0],[0,0]];
test.weights = [[[1,1,0,0]],[[1],[1]]];

console.log(test.train([[-1,-1,1,1]],[[-1,-1]]));

var itterations = 0;
var info1 = document.getElementById('info1');
var info2 = document.getElementById('info2'); 

var allTheError = [];

//learn();


function learn () {
    
    itterations ++;
    info2.innerHTML = itterations;
    
    var newTrainingData = [];
    
    for (var i = 0; i < trainingSetSize; i++) {
        newTrainingData[i] = [getRandomInt(1,99) / 100,getRandomInt(1,99) / 100];
        newTrainingData[i][2] = Math.pow(newTrainingData[i][0],2);
        newTrainingData[i][3] = Math.pow(newTrainingData[i][0],2);
        
    }
    
    
   
    var trainingSolutions = [];
    
    for (var i = 0; i < trainingSetSize; i++) {
        
            trainingSolutions[i] = [];
            var correct = rule(newTrainingData[i][0],newTrainingData[i][1]);
            trainingSolutions[i] = (correct == 1) ? [1,0] : [0,1];
            
    }
    
   
    
    
    var currentError = test.train(newTrainingData,trainingSolutions);
    console.log(currentError);
    allTheError.push(currentError);
    
    var meanError = mean(allTheError);
   
    
    fillCanvasByMethod(ctx2,judge);
    
    
    document.getElementById("info3").innerHTML = test.learningRate;
    intervalSize = 1000 / document.getElementById("interval_size").value;
}
var interval;
var started = false;
var startButton = document.getElementById("start_button");
var expectedCanvas = document.getElementById("expected_canvas");
var actualCanvas = document.getElementById("actual_canvas");
var ctx1 = expectedCanvas.getContext("2d");
var ctx2 = actualCanvas.getContext("2d");



function start() {
    if (started == true) {
        started = false;
        window.clearInterval(interval);
        startButton.innerHTML = "Start";
        
        return;
    }
    started = true;
    startButton.innerHTML = "Stop";
    intervalSize = 1000 / document.getElementById("interval_size").value;
    interval = window.setInterval(learn,intervalSize);
    
}

//fillCanvasByMethod(ctx1,rule);

//Method is a function that returns some output
//Canvas ctx is the context of the canvas to fill

function fillCanvasByMethod (canvasCtx,method) {
    
    let b = 0;
    for (let y = 0; y < 500; y+=5) {
        for (let x = 0; x < 500; x+=5) {
            
            let result = method(x/500,y/500,Math.pow(x/500,2),Math.pow(y/500,2));
            
            
            
            
            canvasCtx.fillStyle = (Math.round(result) == 1) ? "blue" : "red";
            canvasCtx.fillRect(x,y,5,5);
        }
    }
    
    
}

