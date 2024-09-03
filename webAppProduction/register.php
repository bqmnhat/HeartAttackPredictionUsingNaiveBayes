<?php
include('db.php');
$error = ""; // Initialize an empty error message

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $username = $_POST['username'];
    $password = password_hash($_POST['password'], PASSWORD_DEFAULT);

    // Check if the username is already registered
    $checkUserQuery = "SELECT * FROM users WHERE username='$username'";
    $checkUserResult = $conn->query($checkUserQuery);

    if ($checkUserResult->num_rows > 0) {
        // User already exists, display an error message to the user
        http_response_code(409);
        echo "Error: This username is already registered. Please choose a different username.";
    } else {
        // User does not exist, proceed with registration
        $sql = "INSERT INTO users (username, password) VALUES ('$username', '$password')";

        if ($conn->query($sql) === TRUE) {
            // Registration successful, redirect to Diagnosis.html
            echo "Registration successful!";
            header("Location: Diagnosis.html");
            exit();
        } else {
            http_response_code(500);
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
}
?>
