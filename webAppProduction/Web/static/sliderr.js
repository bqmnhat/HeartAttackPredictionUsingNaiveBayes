
var dietr = document.getElementById("dietrange");
var dietd = document.getElementById("dietdisplay");
var dietval = dietr.value;
var value = (dietr.value-dietr.min)/(dietr.max-dietr.min)*100
dietr.style.background = 'linear-gradient(to right, #00A2E8 0%, #00A2E8 ' + value + '%, #fff ' + value + '%, white 100%)'
        
dietd.innerHTML = dietval;
dietr.oninput = function() { 
    dietval = dietr.value;
    dietd.innerHTML = dietval;
    value = (this.value-this.min)/(this.max-this.min)*100
        this.style.background = 'linear-gradient(to right, #00A2E8 0%, #00A2E8 ' + value + '%, #fff ' + value + '%, white 100%)';
}

var stressr = document.getElementById("stressrange");
var stressd = document.getElementById("stressdisplay");
var stressval = stressr.value;
var value = (stressr.value-stressr.min)/(stressr.max-stressr.min)*100
stressr.style.background = 'linear-gradient(to right, #00A2E8 0%, #00A2E8 ' + value + '%, #fff ' + value + '%, white 100%)'

stressd.innerHTML = stressval;    
stressr.oninput = function() { 
    stressval = stressr.value;
    stressd.innerHTML = stressval;
    value = (this.value-this.min)/(this.max-this.min)*100
        this.style.background = 'linear-gradient(to right, #00A2E8 0%, #00A2E8 ' + value + '%, #fff ' + value + '%, white 100%)';
}