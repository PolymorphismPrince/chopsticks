var trainingSetSize = 50;
var currentDiv = "";
function rule (x,y) {
    
    if (x + y > 1) {
        return 0;
    }
    
    return 1;
}


var test = new network();

test.initNet([2,2,1]);
addProperty(trainingSetSize,"Training set size");
line();
//setInterval(learn,1000);
learn();
function learn () {
    
    var newTrainingData = [];
    
    for (var i = 0; i < trainingSetSize; i++) {
        newTrainingData[i] = [getRandomInt(1,99) / 100,getRandomInt(1,99) / 100];
        addProperty(newTrainingData[i],"Training Data " + (i + 1));
        
    }
    line();
    
    var results = test.train(JSON.parse(JSON.stringify(newTrainingData)));
    
    
    var trainingSolutions = [];
    
    for (var i = 0; i < newTrainingData.length; i++) {
        
            trainingSolutions[i] = [];
            
            trainingSolutions[i].push(rule(newTrainingData[i][0],newTrainingData[i][1]));
            
    }
    
   
    
    
    test.backProp([results,trainingSolutions]);
}

function addProperty (val,name) {
    currentDiv += name + ":    " + JSON.stringify(val) + "<br>";
}

function line () {
    
    document.body.innerHTML += "<div style='width: 400px; height: 400px; overflow: scroll;'>" + currentDiv + "</div> ";
}

