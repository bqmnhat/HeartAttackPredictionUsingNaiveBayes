<?php
use LDAP\Result;

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    
    // Collect input data
    $features = $_POST["features"];
    // Calculate BMI using height and weight
    $height = floatval($_POST['Height']); // Assuming you have a height input
    $weight = floatval($_POST['Weight']); // Assuming you have a weight input

    // Calculate BMI
    $bmi = calculateBMI($height, $weight);

    // Add BMI to the features array
    $features['bmi'] = $bmi;

    $featureValues = implode(" ", array_map('escapeshellarg', $features ));
    // Validate and sanitize user inputs as needed

    // Call Python script with input data
    $pythonScript = "app.py"; 
    $command = "python $pythonScript $featureValues";  
    // echo $command;
    $output = shell_exec($command);

    $prediction = json_decode($output, true);

    // Display the prediction or take further actions
    echo $prediction['result'];

    // You can add more details or redirect the user as needed
}

function calculateBMI($height, $weight) {
    $bmi = $weight / (($height / 100) * ($height / 100));
    return $bmi;
}

?>
