var canvas= document.getElementById("canvas");
var ctx = canvas.getContext("2d");

var handImg = new Image();
handImg.src = "./whole_hand.png";
var fingerImgs = [];

for (var i = 0; i < 4; i ++) {
    fingerImgs.push(new Image());
    fingerImgs[i].src = "./finger" + (i + 1).toString() + ".png";
}

canvas.width = 800;
canvas.height = 800;



var players = {};
players[1] = new Player(1);
players[2] = new Player(2);

function Player (num) {
    
    this.number = num;
    
    this.left = new Hand(1,num);
    this.right = new Hand(2,num);
    
    
    
    this.tap = function(h1,h2) {
        
    }
}

function Hand(side,num) {
    this.playerNum = num;
    //Sides: left = 1, right = 2
    this.side = side;
    //Fingers are for a left hand
    this.fingers = [0,0,0,1,0];
}

var imagesToDraw = [fingerImgs[0]];
drawImages();

function drawImages () {
    imagesToDraw.forEach(function (img) {
      ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);  
    })
}