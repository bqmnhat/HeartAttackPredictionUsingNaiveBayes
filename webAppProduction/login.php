<?php
include('db.php');

$response = ""; // Initialize an empty response message

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $username = $_POST['username'];
    $password = $_POST['password'];

    $checkUserQuery = "SELECT * FROM users WHERE username='$username'";
    $checkUserResult = $conn->query($checkUserQuery);

    if ($checkUserResult->num_rows == 1) {
        $row = $checkUserResult->fetch_assoc();
        if (password_verify($password, $row['password'])) {
            // Login successful
            $response = "Login successful!";
            $_SESSION["username"] = $username;
        } else {
            // Incorrect password
            $response = "Incorrect password";
        }
    } else {
        // User not found
        $response = "User not found";
    }

    // Return the response to the JavaScript on the client side
    echo $response;
}
?>
